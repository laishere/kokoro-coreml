import SwiftUI
import Combine

@MainActor
final class BenchmarkViewModel: ObservableObject {
    @Published var data: BenchmarkData?
    @Published var voice: VoicePack?
    @Published var engine: KokoroEngine?
    @Published var loadError: String?

    @Published var results: [Int: SynthResult] = [:]
    @Published var runningCase: Int?  // case being benchmarked right now
    @Published var isRunningAll = false
    @Published var playingCase: Int?

    private let player = AudioPlayer()

    init() {
        player.onFinished = { [weak self] in
            Task { @MainActor in self?.playingCase = nil }
        }
    }

    func loadAll() {
        do {
            let d = try BenchmarkData.load()
            self.data = d
            self.voice = try VoicePack.load(name: d.voice)
        } catch {
            self.loadError = "Data load failed: \(error.localizedDescription)"
            return
        }
        Task.detached {
            do {
                let eng = try await KokoroEngine()
                await MainActor.run { self.engine = eng }
            } catch {
                await MainActor.run { self.loadError = "Models load failed: \(error.localizedDescription)" }
            }
        }
    }

    func runAll() {
        guard let engine, let voice, let data, !isRunningAll else { return }
        isRunningAll = true
        results.removeAll()
        let cases = data.cases
        Task.detached {
            for c in cases {
                await MainActor.run { self.runningCase = c.id }
                do {
                    // 1 warmup + 1 measured run
                    _ = try await engine.synthesize(phonemes: c.phonemes, voice: voice)
                    let r = try await engine.synthesize(phonemes: c.phonemes, voice: voice)
                    await MainActor.run { self.results[c.id] = r }
                } catch {
                    await MainActor.run {
                        self.loadError = "Case \(c.id) failed: \(error.localizedDescription)"
                    }
                    break
                }
            }
            await MainActor.run {
                self.runningCase = nil
                self.isRunningAll = false
            }
        }
    }

    func play(_ caseId: Int) {
        guard let result = results[caseId] else { return }
        do {
            try player.play(samples: result.audio)
            playingCase = caseId
        } catch {
            loadError = "Play failed: \(error.localizedDescription)"
        }
    }

    func stopPlay() {
        player.stop()
        playingCase = nil
    }

    var meanSpeed: Double? {
        let speeds = results.values.map { $0.speed }
        guard !speeds.isEmpty else { return nil }
        return speeds.reduce(0, +) / Double(speeds.count)
    }
}

struct BenchmarkView: View {
    @StateObject private var vm = BenchmarkViewModel()

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("CoreML Benchmark")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { runButton }
        }
        .onAppear { if vm.data == nil { vm.loadAll() } }
    }

    @ViewBuilder
    private var content: some View {
        if let err = vm.loadError {
            VStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundStyle(.orange)
                Text(err)
                    .multilineTextAlignment(.center)
                    .padding()
            }
        } else if let data = vm.data {
            ScrollView {
                VStack(spacing: 16) {
                    if vm.engine == nil {
                        loadingBanner
                    }
                    if !vm.results.isEmpty {
                        summaryTable(cases: data.cases)
                    }
                    ForEach(data.cases) { c in
                        caseCard(c)
                    }
                }
                .padding()
            }
        } else {
            ProgressView("Loading…")
        }
    }

    private var loadingBanner: some View {
        HStack {
            ProgressView()
            Text("Loading models…").foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    @ToolbarContentBuilder
    private var runButton: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Button {
                vm.runAll()
            } label: {
                if vm.isRunningAll {
                    ProgressView()
                } else {
                    Label("Run All", systemImage: "play.circle.fill")
                }
            }
            .disabled(vm.engine == nil || vm.isRunningAll)
        }
    }

    // MARK: - Summary table

    @ViewBuilder
    private func summaryTable(cases: [BenchmarkCase]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Summary").font(.headline)

            VStack(spacing: 4) {
                summaryRow(["T_enc", "T_a", "audio", "chain", "speed"], isHeader: true)
                Divider()
                ForEach(cases) { c in
                    if let r = vm.results[c.id] {
                        summaryRow([
                            "\(r.tEnc)",
                            "\(r.tA)",
                            String(format: "%.2fs", r.audioSec),
                            String(format: "%.0fms", r.stages.totalMs),
                            String(format: "%.1fx", r.speed),
                        ], isHeader: false)
                    } else if vm.runningCase == c.id {
                        summaryRow(["\(c.n_phonemes + 2)", "—", "—", "—", "running…"], isHeader: false)
                    } else {
                        summaryRow(["\(c.n_phonemes + 2)", "—", "—", "—", "—"], isHeader: false, dim: true)
                    }
                }
                if let mean = vm.meanSpeed {
                    Divider()
                    HStack {
                        Spacer()
                        Text(String(format: "Mean speed: %.1fx", mean))
                            .font(.caption.bold())
                    }
                }
            }
            .font(.caption.monospacedDigit())
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func summaryRow(_ cells: [String], isHeader: Bool, dim: Bool = false) -> some View {
        HStack {
            ForEach(Array(cells.enumerated()), id: \.offset) { _, s in
                Text(s)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .fontWeight(isHeader ? .semibold : .regular)
        .foregroundStyle(isHeader ? .primary : (dim ? .secondary : .primary))
    }

    // MARK: - Case card

    @ViewBuilder
    private func caseCard(_ c: BenchmarkCase) -> some View {
        let result = vm.results[c.id]
        let isRunning = vm.runningCase == c.id
        let isPlaying = vm.playingCase == c.id

        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Case \(c.id + 1)")
                    .font(.headline)
                Spacer()
                Text("T_enc=\(c.n_phonemes + 2)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Text(c.text)
                .font(.body)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            statsLine(c: c, result: result, isRunning: isRunning)

            if result != nil {
                HStack {
                    Spacer()
                    if isPlaying {
                        Button {
                            vm.stopPlay()
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                    } else {
                        Button {
                            vm.play(c.id)
                        } label: {
                            Label("Play", systemImage: "play.fill")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private func statsLine(c: BenchmarkCase, result: SynthResult?, isRunning: Bool) -> some View {
        if isRunning {
            HStack(spacing: 6) {
                ProgressView().scaleEffect(0.7)
                Text("Synthesizing…").font(.caption).foregroundStyle(.secondary)
            }
        } else if let r = result {
            HStack(spacing: 16) {
                stat("audio", String(format: "%.2fs", r.audioSec))
                stat("synth", String(format: "%.0fms", r.stages.totalMs))
                stat("speed", String(format: "%.1fx", r.speed))
            }
        } else {
            Text("Not yet synthesized").font(.caption).foregroundStyle(.secondary)
        }
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.callout.monospacedDigit())
        }
    }
}

#Preview {
    BenchmarkView()
}
