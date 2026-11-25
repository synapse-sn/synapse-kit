/**
 * Synapse SNN-RAG 프로덕션 신경망 HTTP 어댑터 모듈
 *
 * 지원 규격:
 * 1. OpenAI 호환 임베딩 API (/v1/embeddings — OpenAI, vLLM, TEI, LM Studio, FastAPI 등)
 * 2. Ollama 임베딩 API (/api/embed)
 * 3. Cohere / Jina / TEI 호환 Cross-Encoder 리랭커 API (/rerank, /v1/rerank)
 */

import type { VectorEmbedderFn } from './ai-vector';
import type { NeuralRerankCandidate, NeuralRerankerFn } from './snn-rag';

export const SNN_NEURAL_ADAPTER_CONFIG = {
  DEFAULT_TIMEOUT_MS: 15_000,
  DEFAULT_BATCH_SIZE: 32,
  DEFAULT_OPENAI_EMBED_PATH: '/v1/embeddings',
  DEFAULT_OLLAMA_EMBED_PATH: '/api/embed',
  DEFAULT_RERANK_PATH: '/rerank',
} as const;

export type BatchVectorEmbedderFn = (texts: readonly string[]) => Promise<number[][]>;

export interface OpenAiEmbedderOptions {
  readonly baseUrl: string;
  readonly model: string;
  readonly apiKey?: string;
  readonly endpointPath?: string;
  readonly timeoutMs?: number;
  readonly batchSize?: number;
  /** 비대칭 임베딩 모델(예: E5, BGE)용 입력 접두사 */
  readonly inputPrefix?: string;
  readonly headers?: Readonly<Record<string, string>>;
}

export interface OllamaEmbedderOptions {
  readonly baseUrl: string;
  readonly model: string;
  readonly endpointPath?: string;
  readonly timeoutMs?: number;
  readonly inputPrefix?: string;
}

export interface HttpRerankerOptions {
  readonly baseUrl: string;
  readonly model?: string;
  readonly apiKey?: string;
  readonly endpointPath?: string;
  readonly timeoutMs?: number;
  /** 로짓(음수~양수) 반환 모델인 경우 시그모이드 정규화 적용 여부 */
  readonly applySigmoid?: boolean;
  readonly headers?: Readonly<Record<string, string>>;
}

function joinEndpointUrl(baseUrl: string, path: string): string {
  const trimmedBase = baseUrl.replace(/\/+$/, '');
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${trimmedBase}${normalizedPath}`;
}

function sigmoid(x: number): number {
  if (x >= 20) return 1;
  if (x <= -20) return 0;
  return 1 / (1 + Math.exp(-x));
}

/**
 * OpenAI 호환 배치 및 단건 임베더 생성기
 */
export function createOpenAiCompatibleEmbedder(options: OpenAiEmbedderOptions): {
  readonly embed: VectorEmbedderFn;
  readonly embedBatch: BatchVectorEmbedderFn;
} {
  if (!options.baseUrl || !options.model) {
    throw new Error('임베딩 어댑터 초기화에는 baseUrl과 model 설정이 필요합니다.');
  }

  const url = joinEndpointUrl(
    options.baseUrl,
    options.endpointPath ?? SNN_NEURAL_ADAPTER_CONFIG.DEFAULT_OPENAI_EMBED_PATH
  );
  const timeoutMs = Math.max(500, options.timeoutMs ?? SNN_NEURAL_ADAPTER_CONFIG.DEFAULT_TIMEOUT_MS);
  const batchSize = Math.max(1, options.batchSize ?? SNN_NEURAL_ADAPTER_CONFIG.DEFAULT_BATCH_SIZE);
  const prefix = options.inputPrefix ?? '';

  const requestChunk = async (chunk: readonly string[]): Promise<number[][]> => {
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      };
      if (options.apiKey) {
        headers.Authorization = `Bearer ${options.apiKey}`;
      }

      const prefixedInputs = prefix ? chunk.map((t) => `${prefix}${t}`) : chunk;
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: options.model,
          input: prefixedInputs,
        }),
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} (${response.statusText})`);
      }

      const payload = (await response.json()) as {
        data?: Array<{ index?: number; embedding?: number[] }>;
        embeddings?: number[][];
      };

      if (Array.isArray(payload.data) && payload.data.length === chunk.length) {
        const sorted = [...payload.data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
        return sorted.map((item) => {
          if (!Array.isArray(item.embedding)) {
            throw new Error('응답 항목에 유효한 embedding 배열이 없습니다.');
          }
          return item.embedding;
        });
      }

      if (Array.isArray(payload.embeddings) && payload.embeddings.length === chunk.length) {
        return payload.embeddings;
      }

      throw new Error('임베딩 응답 형식이 OpenAI 호환 규격과 일치하지 않습니다.');
    } catch (err) {
      throw new Error(
        `OpenAI 호환 임베딩 요청 실패 (${url}): ${err instanceof Error ? err.message : String(err)}`
      );
    }
  };

  const embedBatch: BatchVectorEmbedderFn = async (texts) => {
    if (texts.length === 0) return [];
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += batchSize) {
      const chunk = texts.slice(i, i + batchSize);
      const vectors = await requestChunk(chunk);
      out.push(...vectors);
    }
    return out;
  };

  const embed: VectorEmbedderFn = async (text) => {
    const [vec] = await requestChunk([text]);
    return vec;
  };

  return { embed, embedBatch };
}

/**
 * Ollama (/api/embed) 호환 임베더 생성기
 */
export function createOllamaEmbedder(options: OllamaEmbedderOptions): {
  readonly embed: VectorEmbedderFn;
  readonly embedBatch: BatchVectorEmbedderFn;
} {
  if (!options.baseUrl || !options.model) {
    throw new Error('Ollama 임베딩 어댑터 초기화에는 baseUrl과 model 설정이 필요합니다.');
  }

  const url = joinEndpointUrl(
    options.baseUrl,
    options.endpointPath ?? SNN_NEURAL_ADAPTER_CONFIG.DEFAULT_OLLAMA_EMBED_PATH
  );
  const timeoutMs = Math.max(500, options.timeoutMs ?? SNN_NEURAL_ADAPTER_CONFIG.DEFAULT_TIMEOUT_MS);
  const prefix = options.inputPrefix ?? '';

  const embedBatch: BatchVectorEmbedderFn = async (texts) => {
    if (texts.length === 0) return [];
    try {
      const input = prefix ? texts.map((t) => `${prefix}${t}`) : texts;
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: options.model, input }),
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} (${response.statusText})`);
      }

      const payload = (await response.json()) as { embeddings?: number[][]; embedding?: number[] };
      if (Array.isArray(payload.embeddings)) {
        return payload.embeddings;
      }
      if (texts.length === 1 && Array.isArray(payload.embedding)) {
        return [payload.embedding];
      }
      throw new Error('Ollama 임베딩 응답에 유효한 embeddings 필드가 없습니다.');
    } catch (err) {
      throw new Error(
        `Ollama 임베딩 요청 실패 (${url}): ${err instanceof Error ? err.message : String(err)}`
      );
    }
  };

  const embed: VectorEmbedderFn = async (text) => {
    const [vec] = await embedBatch([text]);
    return vec;
  };

  return { embed, embedBatch };
}

/**
 * Cohere / Jina / HuggingFace TEI 호환 Cross-Encoder HTTP 리랭커 생성기
 */
export function createHttpReranker(options: HttpRerankerOptions): NeuralRerankerFn {
  if (!options.baseUrl) {
    throw new Error('HTTP 리랭커 어댑터 초기화에는 baseUrl 설정이 필요합니다.');
  }

  const url = joinEndpointUrl(
    options.baseUrl,
    options.endpointPath ?? SNN_NEURAL_ADAPTER_CONFIG.DEFAULT_RERANK_PATH
  );
  const timeoutMs = Math.max(500, options.timeoutMs ?? SNN_NEURAL_ADAPTER_CONFIG.DEFAULT_TIMEOUT_MS);
  const useSigmoid = options.applySigmoid ?? false;

  return async (query: string, candidates: readonly NeuralRerankCandidate[]) => {
    if (candidates.length === 0) return {};

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      };
      if (options.apiKey) {
        headers.Authorization = `Bearer ${options.apiKey}`;
      }

      const documents = candidates.map((c) => (c.title ? `${c.title}\n${c.text}` : c.text));
      const body: Record<string, unknown> = {
        query,
        documents,
        texts: documents,
        top_n: candidates.length,
      };
      if (options.model) {
        body.model = options.model;
      }

      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} (${response.statusText})`);
      }

      const payload = (await response.json()) as
        | Array<{ index?: number; score?: number; relevance_score?: number }>
        | {
            results?: Array<{ index?: number; relevance_score?: number; score?: number }>;
            scores?: number[];
          };

      const scoreMap: Record<string, number> = {};
      const normalize = (raw: number): number => {
        if (!Number.isFinite(raw)) return 0;
        if (useSigmoid || raw < 0 || raw > 1) return sigmoid(raw);
        return Math.max(0, Math.min(1, raw));
      };

      // 1) HuggingFace TEI 형식: [{ index, score }, ...]
      if (Array.isArray(payload)) {
        for (const item of payload) {
          const idx = item.index ?? -1;
          const raw = item.relevance_score ?? item.score ?? 0;
          if (idx >= 0 && idx < candidates.length) {
            scoreMap[candidates[idx].id] = normalize(raw);
          }
        }
        return scoreMap;
      }

      // 2) Cohere / Jina 형식: { results: [{ index, relevance_score }, ...] }
      if (Array.isArray(payload.results)) {
        for (const item of payload.results) {
          const idx = item.index ?? -1;
          const raw = item.relevance_score ?? item.score ?? 0;
          if (idx >= 0 && idx < candidates.length) {
            scoreMap[candidates[idx].id] = normalize(raw);
          }
        }
        return scoreMap;
      }

      // 3) 단순 배열 형식: { scores: [number, ...] }
      if (Array.isArray(payload.scores) && payload.scores.length === candidates.length) {
        for (let i = 0; i < candidates.length; i++) {
          scoreMap[candidates[i].id] = normalize(payload.scores[i]);
        }
        return scoreMap;
      }

      throw new Error('리랭커 응답 형식을 해석할 수 없습니다.');
    } catch (err) {
      throw new Error(
        `HTTP 리랭커 요청 실패 (${url}): ${err instanceof Error ? err.message : String(err)}`
      );
    }
  };
}
