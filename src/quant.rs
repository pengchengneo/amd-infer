//! IQ4_XS wire format: 256 values / 136 bytes. GGML non-linear codebook.
pub const IQ4_VALUES: [f32; 16] = [
    -127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89., 113.,
];
pub fn half(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = (h >> 10) & 31;
    let m = (h & 1023) as u32;
    if e == 0 {
        if m == 0 {
            return f32::from_bits(sign);
        }
        let mut x = m;
        let mut exp = 113u32;
        while x & 1024 == 0 {
            x <<= 1;
            exp -= 1
        }
        f32::from_bits(sign | (exp << 23) | ((x & 1023) << 13))
    } else if e == 31 {
        f32::from_bits(sign | 0x7f800000 | (m << 13))
    } else {
        f32::from_bits(sign | (((e as u32) + 112) << 23) | (m << 13))
    }
}
pub fn iq4_xs(block: &[u8], out: &mut [f32]) -> Result<(), &'static str> {
    if block.len() != 136 || out.len() != 256 {
        return Err("IQ4_XS block/output length");
    }
    let d = half(u16::from_le_bytes([block[0], block[1]]));
    let hi = u16::from_le_bytes([block[2], block[3]]);
    for s in 0..8 {
        let lo = (block[4 + s / 2] >> (4 * (s % 2))) & 15;
        let scale = (lo as i32 | (((hi >> (2 * s)) & 3) as i32) << 4) - 32;
        let ds = d * scale as f32;
        for j in 0..16 {
            let q = block[8 + s * 16 + j];
            out[s * 32 + j] = ds * IQ4_VALUES[(q & 15) as usize];
            out[s * 32 + j + 16] = ds * IQ4_VALUES[(q >> 4) as usize];
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_edges() {
        assert_eq!(half(0x3c00), 1.);
        assert_eq!(half(0xbc00), -1.);
        assert_eq!(half(1), 2f32.powi(-24));
        assert!(half(0x7c00).is_infinite());
        assert!(half(0x7e00).is_nan());
    }
    #[test]
    fn iq4_scale_and_nibble_order() {
        let mut b = [0u8; 136];
        b[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        b[2..4].copy_from_slice(&0xaaaau16.to_le_bytes());
        b[4..8].fill(0x11);
        b[8..].fill(0xf0);
        let mut y = [0.; 256];
        iq4_xs(&b, &mut y).unwrap();
        assert_eq!(y[0], -127.);
        assert_eq!(y[16], 113.);
        assert_eq!(y[32], -127.);
    }
}
