import SwiftUI

struct ContentView: View {
    private let featureNames = [
        "passenger_capacity", "design_range_km", "cruise_speed_kmh",
        "battery_specific_energy_wh_per_kg", "hydrogen_storage_efficiency",
        "hybridization_ratio"
    ]
    @State private var values = [4.0, 250.0, 180.0, 300.0, 0.65, 0.5]
    @State private var category = "conventional_turboprop"
    @State private var predictions: [Prediction] = []
    @State private var message = "Run the bundled Core ML model and capture cold + warm evidence."
    @State private var inputWarning: String?
    @State private var evidence: IOSBenchmarkEvidence?
    @State private var evidenceURL: URL?
    @State private var isRunning = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Aircraft design — generated-data model") {
                    ForEach(featureNames.indices, id: \.self) { index in
                        TextField(featureNames[index].replacingOccurrences(of: "_", with: " "), value: $values[index], format: .number)
                            .keyboardType(.decimalPad)
                    }
                    Picker("Propulsion architecture", selection: $category) {
                        ForEach(["conventional_turboprop", "fuel_cell_electric", "parallel_hybrid", "series_hybrid"], id: \.self) { value in
                            Text(value.replacingOccurrences(of: "_", with: " ")).tag(value)
                        }
                    }
                    Button("Reset to reference inputs", action: resetInputs)
                    if let inputWarning {
                        Label(inputWarning, systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }
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
            .disabled(isRunning)
            .onChange(of: values) { _, _ in updateInputWarning() }
        }
        .task { updateInputWarning() }
    }

    private func runBenchmark() {
        guard values.count == featureNames.count, values.allSatisfy(\.isFinite) else {
            message = "Enter finite numeric values for every feature."
            return
        }
        updateInputWarning()
        isRunning = true
        evidence = nil
        evidenceURL = nil
        let inputValues = values
        let inputCategory = category
        Task {
            do {
                let result = try await Task.detached(priority: .userInitiated) {
                    try IOSBenchmarkRunner.run(numericValues: inputValues, category: inputCategory)
                }.value
                evidence = result
                predictions = result.outputs.map { Prediction(name: $0.name, value: $0.value) }
                evidenceURL = try result.writeTemporaryJSON()
                message = "Completed 1 model-load + first-prediction and \(result.latency.warmRuns) warm runs."
            } catch {
                predictions = []
                message = error.localizedDescription
            }
            isRunning = false
        }
    }

    private func resetInputs() {
        values = [65.0, 950.0, 535.0, 527.0, 0.57, 0.24]
        category = "conventional_turboprop"
        message = "Reference inputs restored."
        updateInputWarning()
    }

    private func updateInputWarning() {
        let means = [64.93524, 950.4023, 534.7646, 526.7014, 0.57418, 0.24301]
        let scales = [14.40655, 318.0289, 66.45373, 130.1412, 0.072085, 0.215283]
        guard values.count == means.count, values.allSatisfy(\.isFinite) else {
            inputWarning = "Some inputs are not finite."
            return
        }
        let maximumZ = zip(values, zip(means, scales)).map { pair in
            abs((pair.0 - pair.1.0) / pair.1.1)
        }.max() ?? 0
        inputWarning = maximumZ > 3 ? "Inputs are outside the training distribution (max |z| = \(maximumZ.formatted(.number.precision(.fractionLength(1))))." : nil
    }
}
