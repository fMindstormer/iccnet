// What macOS does with a profile and a PNG that carries it.
// usage: colorsync <profile.icc> <image.png> <out prefix>   (swiftc -O probe/colorsync.swift -o colorsync)
//   prints ColorSync's verification result and whether CoreGraphics accepts the profile,
//   writes <prefix>.cg.f32 (ImageIO decode drawn into a float sRGB context, as apps draw)
//   and <prefix>.cs.f32 (direct ColorSync float transform, relative colorimetric, of the
//   image's own pixels), both raw little-endian float RGBA.
import ColorSync
import CoreGraphics
import Foundation
import ImageIO

let args = CommandLine.arguments
let iccData = try! Data(contentsOf: URL(fileURLWithPath: args[1]))

var err: Unmanaged<CFError>?
if let prof = ColorSyncProfileCreate(iccData as CFData, &err)?.takeRetainedValue() {
    var errors: Unmanaged<CFError>?
    var warnings: Unmanaged<CFError>?
    let ok = ColorSyncProfileVerify(prof, &errors, &warnings)
    print("ColorSyncProfileVerify:", ok)
    if let e = errors?.takeRetainedValue() { print("  errors:", e) }
    if let w = warnings?.takeRetainedValue() { print("  warnings:", w) }
} else {
    print("ColorSyncProfileCreate failed:", err?.takeRetainedValue() as Any)
}
print("CGColorSpace(iccData:):", CGColorSpace(iccData: iccData as CFData) != nil ? "accepted" : "rejected")

let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[2]) as CFURL, nil)!
let img = CGImageSourceCreateImageAtIndex(src, 0, nil)!
print("decoded image color space:", img.colorSpace.map { String(describing: $0.name ?? "unnamed" as CFString) } ?? "none")
let w = img.width, h = img.height
let srgb = CGColorSpace(name: CGColorSpace.sRGB)!

// 1. CoreGraphics: draw into a float sRGB bitmap, the way an app renders the image
let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.floatComponents.rawValue
    | CGBitmapInfo.byteOrder32Little.rawValue
let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 32, bytesPerRow: w * 16,
                    space: srgb, bitmapInfo: info)!
ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
try! Data(bytes: ctx.data!, count: w * h * 16).write(to: URL(fileURLWithPath: args[3] + ".cg.f32"))

// 2. ColorSync directly: profile -> sRGB on float pixels, no CoreGraphics caching
guard let prof = ColorSyncProfileCreate(iccData as CFData, nil)?.takeRetainedValue(),
      let dst = ColorSyncProfileCreateWithName(kColorSyncSRGBProfile.takeUnretainedValue())?.takeRetainedValue()
else { print("no ColorSync profile; skipping direct transform"); exit(0) }
let seq: [[String: Any]] = [
    [kColorSyncProfile.takeUnretainedValue() as String: prof,
     kColorSyncRenderingIntent.takeUnretainedValue() as String: kColorSyncRenderingIntentRelative.takeUnretainedValue(),
     kColorSyncTransformTag.takeUnretainedValue() as String: kColorSyncTransformDeviceToPCS.takeUnretainedValue()],
    [kColorSyncProfile.takeUnretainedValue() as String: dst,
     kColorSyncRenderingIntent.takeUnretainedValue() as String: kColorSyncRenderingIntentRelative.takeUnretainedValue(),
     kColorSyncTransformTag.takeUnretainedValue() as String: kColorSyncTransformPCSToDevice.takeUnretainedValue()],
]
guard let xf = ColorSyncTransformCreate(seq as CFArray, nil)?.takeRetainedValue() else {
    print("ColorSyncTransformCreate failed"); exit(0)
}
// the image's own decoded 8-bit pixels (no color matching) as float RGBA input
let bytes = img.dataProvider!.data! as Data
let bpp = img.bitsPerPixel / 8
var fin = [Float](repeating: 1, count: w * h * 4)
for y in 0..<h { for x in 0..<w { for c in 0..<3 {
    fin[(y * w + x) * 4 + c] = Float(bytes[y * img.bytesPerRow + x * bpp + c]) / 255
} } }
var fout = [Float](repeating: 0, count: w * h * 4)
let ok = fin.withUnsafeBytes { s in fout.withUnsafeMutableBytes { d in
    ColorSyncTransformConvert(xf, w, h, d.baseAddress!, kColorSync32BitFloat,
                              ColorSyncDataLayout(CGImageAlphaInfo.last.rawValue), w * 16,
                              s.baseAddress!, kColorSync32BitFloat,
                              ColorSyncDataLayout(CGImageAlphaInfo.last.rawValue), w * 16, nil)
} }
print("ColorSyncTransformConvert:", ok)
fout.withUnsafeBytes { try! Data($0).write(to: URL(fileURLWithPath: args[3] + ".cs.f32")) }
