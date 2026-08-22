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
    @State private var message = "Export and add the model resources, then run a design point."

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
                Section { Button("Run on device", action: runPrediction) }
                Section("Result") {
                    Text(message)
                    ForEach(predictions) { prediction in
                        LabeledContent(prediction.name, value: prediction.value.formatted(.number.precision(.fractionLength(3))))
                    }
                }
            }
            .navigationTitle("EdgeGenBench")
        }
    }

    private func runPrediction() {
        do {
            let predictor = try SurrogatePredictor()
            predictions = try predictor.predict(numericValues: values, category: category)
            message = "Inference completed with Core ML."
        } catch {
            predictions = []
            message = error.localizedDescription
        }
    }
}

