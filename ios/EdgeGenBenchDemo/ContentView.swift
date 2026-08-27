import SwiftUI

struct ContentView: View {
    private let featureNames = [
        "passenger_capacity", "design_range_km", "cruise_speed_kmh",
        "battery_specific_energy_wh_per_kg", "hydrogen_storage_efficiency",
        "hybridization_ratio"
    ]
    @State private var values = [4.0, 250.0, 180.0, 300.0, 0.65, 0.5]
    @State private var category = "battery_electric"
    @State private var predictions: [Prediction] = []
    @State private var message = "Run the bundled Core ML model and capture cold + warm evidence."
    @State private var evidence: IOSBenchmarkEvidence?
    @State private var evidenceURL: URL?
    @State private var isRunning = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Aircraft design") {
                    ForEach(featureNames.indices, id: \.self) { index in
                        TextField(featureNames[index], value: $values[index], format: .number)
                            .keyboardType(.decimalPad)
                    }
                    TextField("propulsion_architecture", text: $category)
                }
                Section {
                    Button(isRunning ? "Benchmarking…" : "Run cold + warm benchmark", action: runBenchmark)
                        .disabled(isRunning)
                    if let evidenceURL {
                        ShareLink(item: evidenceURL) {
                            Label("Export evidence JSON", systemImage: "square.and.arrow.up")
                        }
                    }
                }
                Section("Result") {
                    Text(message)
                    ForEach(predictions) { prediction in
                        LabeledContent(prediction.name, value: prediction.value.formatted(.number.precision(.fractionLength(3))))
                    }
                    if let evidence {
                        LabeledContent("Backend", value: evidence.backend)
                        LabeledContent("Cold", value: "\(evidence.latency.coldMs.formatted(.number.precision(.fractionLength(3)))) ms")
                        LabeledContent("Warm mean", value: "\(evidence.latency.warmMeanMs.formatted(.number.precision(.fractionLength(3)))) ms")
                        LabeledContent("Warm p95", value: "\(evidence.latency.warmP95Ms.formatted(.number.precision(.fractionLength(3)))) ms")
                        Text("ANE placement and power are not inferred; use Instruments for those claims.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("EdgeGenBench")
        }
    }

    private func runBenchmark() {
        isRunning = true
        do {
            let result = try IOSBenchmarkRunner.run(numericValues: values, category: category)
            evidence = result
            predictions = result.outputs.map { Prediction(name: $0.name, value: $0.value) }
            evidenceURL = try result.writeTemporaryJSON()
            message = "Core ML benchmark completed (1 cold + \(result.latency.warmRuns) warm runs)."
        } catch {
            predictions = []
            evidence = nil
            evidenceURL = nil
            message = error.localizedDescription
        }
        isRunning = false
    }
}
