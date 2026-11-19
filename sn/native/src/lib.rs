//! sn 제어 언어(Synapse Control Language)용 Rust 네이티브 C ABI 코어
//! - Bun FFI(`bun:ffi`) 및 외부 네이티브 프로그램이 직접 호출할 수 있는 `extern "C"` 심볼을 제공합니다.

use std::slice;

const MAX_HASH_LEN: i32 = 64 * 1024 * 1024;

/// FNV-1a 64비트 해시 (초고속 비암호화 무결성 체크섬)
///
/// # Safety
/// `ptr`이 null이 아니고 최소 `len` 바이트 이상의 유효한 메모리를 가리켜야 합니다.
#[no_mangle]
pub unsafe extern "C" fn sn_fast_hash(ptr: *const u8, len: i32) -> i64 {
    if ptr.is_null() || len <= 0 || len > MAX_HASH_LEN {
        return 0;
    }
    let bytes = unsafe { slice::from_raw_parts(ptr, len as usize) };
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash as i64
}

/// 네이티브 기계어 점수 게이트: `0 <= score <= limit` 이면 1, 아니면 0 반환
#[no_mangle]
pub extern "C" fn sn_score_gate(score: i32, limit: i32) -> i32 {
    if score >= 0 && score <= limit {
        1
    } else {
        0
    }
}

/// 네이티브 가중치 계산: `(a * 2) + b`
#[no_mangle]
pub extern "C" fn sn_weighted_sum(a: i32, b: i32) -> i32 {
    a.saturating_mul(2).saturating_add(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sn_score_gate() {
        assert_eq!(sn_score_gate(0, 100), 1);
        assert_eq!(sn_score_gate(85, 100), 1);
        assert_eq!(sn_score_gate(-1, 100), 0);
        assert_eq!(sn_score_gate(101, 100), 0);
    }

    #[test]
    fn test_sn_weighted_sum() {
        assert_eq!(sn_weighted_sum(15, 7), 37);
        assert_eq!(sn_weighted_sum(i32::MAX, 1), i32::MAX);
    }

    #[test]
    fn test_sn_fast_hash_deterministic() {
        let data = b"synapse-sn-control-plane";
        let h1 = unsafe { sn_fast_hash(data.as_ptr(), data.len() as i32) };
        let h2 = unsafe { sn_fast_hash(data.as_ptr(), data.len() as i32) };
        assert_ne!(h1, 0);
        assert_eq!(h1, h2);

        let empty = unsafe { sn_fast_hash(std::ptr::null(), 0) };
        assert_eq!(empty, 0);
    }
}
