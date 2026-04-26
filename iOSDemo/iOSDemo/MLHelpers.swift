import CoreML
import Accelerate

func makeFloatArray(_ values: [Float], shape: [Int]) -> MLMultiArray {
    let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    let ptr = a.dataPointer.bindMemory(to: Float.self, capacity: values.count)
    values.withUnsafeBufferPointer { ptr.initialize(from: $0.baseAddress!, count: values.count) }
    return a
}

func makeFloat16Array(_ values: [Float], shape: [Int]) -> MLMultiArray {
    let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    let ptr = a.dataPointer.bindMemory(to: UInt16.self, capacity: values.count)
    var src = values
    src.withUnsafeMutableBufferPointer { srcBuf in
        let srcPtr = UnsafeRawPointer(srcBuf.baseAddress!).assumingMemoryBound(to: Float.self)
        let dstPtr = UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: UInt16.self)
        var bufferSrc = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr),
                                       height: 1, width: vImagePixelCount(values.count),
                                       rowBytes: values.count * 4)
        var bufferDst = vImage_Buffer(data: UnsafeMutableRawPointer(dstPtr),
                                       height: 1, width: vImagePixelCount(values.count),
                                       rowBytes: values.count * 2)
        vImageConvert_PlanarFtoPlanar16F(&bufferSrc, &bufferDst, 0)
    }
    return a
}

func makeInt32Array(_ values: [Int32], shape: [Int]) -> MLMultiArray {
    let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
    let ptr = a.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
    values.withUnsafeBufferPointer { ptr.initialize(from: $0.baseAddress!, count: values.count) }
    return a
}

func toFloats(_ a: MLMultiArray) -> [Float] {
    let n = a.count
    if a.dataType == .float32 {
        let p = a.dataPointer.bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
    if a.dataType == .float16 {
        let srcPtr = a.dataPointer.bindMemory(to: UInt16.self, capacity: n)
        var result = [Float](repeating: 0, count: n)
        result.withUnsafeMutableBufferPointer { dstBuf in
            var bufferSrc = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr),
                                           height: 1, width: vImagePixelCount(n),
                                           rowBytes: n * 2)
            var bufferDst = vImage_Buffer(data: UnsafeMutableRawPointer(dstBuf.baseAddress!),
                                           height: 1, width: vImagePixelCount(n),
                                           rowBytes: n * 4)
            vImageConvert_Planar16FtoPlanarF(&bufferSrc, &bufferDst, 0)
        }
        return result
    }
    var result = [Float](repeating: 0, count: n)
    for i in 0..<n { result[i] = a[i].floatValue }
    return result
}
