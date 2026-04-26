import Foundation

struct BenchmarkCase: Decodable, Identifiable {
    let id: Int
    let text: String
    let phonemes: String
    let n_phonemes: Int
}

struct BenchmarkData: Decodable {
    let voice: String
    let lang: String
    let sample_rate: Int
    let cases: [BenchmarkCase]

    static func load() throws -> BenchmarkData {
        guard let url = Bundle.main.url(forResource: "benchmark_data", withExtension: "json") else {
            throw NSError(domain: "BenchmarkData", code: 1, userInfo: [NSLocalizedDescriptionKey: "benchmark_data.json not in bundle"])
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(BenchmarkData.self, from: data)
    }
}

struct VoicePack {
    let name: String
    let data: [Float]  // [510, 256] flat

    static func load(name: String) throws -> VoicePack {
        guard let url = Bundle.main.url(forResource: name, withExtension: "bin") else {
            throw NSError(domain: "VoicePack", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(name).bin not in bundle"])
        }
        let bytes = try Data(contentsOf: url)
        let count = bytes.count / MemoryLayout<Float>.size
        var floats = [Float](repeating: 0, count: count)
        _ = floats.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
        return VoicePack(name: name, data: floats)
    }

    /// Style s embedding [128] (second half of voice vector at row T-1).
    func styleS(forTokenCount T: Int) -> [Float] {
        let idx = min(max(T - 1, 0), 509)
        let start = idx * 256 + 128
        return Array(data[start..<start + 128])
    }

    /// Style timbre embedding [128] (first half of voice vector at row T-1).
    func styleTimbre(forTokenCount T: Int) -> [Float] {
        let idx = min(max(T - 1, 0), 509)
        let start = idx * 256
        return Array(data[start..<start + 128])
    }
}

struct Vocab {
    let map: [String: Int]

    static func load() throws -> Vocab {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "json") else {
            throw NSError(domain: "Vocab", code: 1, userInfo: [NSLocalizedDescriptionKey: "vocab.json not in bundle"])
        }
        let data = try Data(contentsOf: url)
        let map = try JSONSerialization.jsonObject(with: data) as! [String: Int]
        return Vocab(map: map)
    }

    /// Convert phoneme string to token ids, wrapped with BOS/EOS (0).
    func tokens(for phonemes: String) -> [Int32] {
        var ids: [Int32] = [0]
        for ch in phonemes {
            if let id = map[String(ch)] { ids.append(Int32(id)) }
        }
        ids.append(0)
        return ids
    }
}
