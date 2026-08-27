import Foundation
import UIKit

struct LatencySummary: Codable {
    let coldMs: Double
    let warmMeanMs: Double
    let warmP95Ms: Double
    let warmRuns: Int
}

struct IOSDeviceIdentity: Codable {
    let model: String
    let systemName: String
    let systemVersion: String
    let simulator: Bool
}

struct IOSBenchmarkEvidence: Codable {
    let schemaVersion: String
    let capturedAtUTC: String
    let appVersion: String
    let backend: String
    let requestedComputeUnits: String
    let neuralEnginePlacement: String
    let powerMeasurement: String
    let thermalStateBefore: String
    let thermalStateAfter: String
    let lowPowerMode: Bool
    let sourceModelSha256: String
    let preprocessingSha256: String
    let contractSha256: String
    let device: IOSDeviceIdentity
    let latency: LatencySummary
    let outputMaxAbsDrift: Double
    let outputs: [PredictionValue]
}

struct PredictionValue: Codable {
    let name: String
    let value: Double
}

enum BenchmarkStatistics {
    static func mean(_ values: [Double]) -> Double {
        values.reduce(0, +) / Double(values.count)
    }

    static func percentile95(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let index = min(sorted.count - 1, Int(ceil(Double(sorted.count) * 0.95)) - 1)
        return sorted[index]
    }
}

enum IOSBenchmarkRunner {
    static func run(numericValues: [Double], category: String, warmRuns: Int = 100) throws -> IOSBenchmarkEvidence {
        precondition(warmRuns > 0)
        let thermalBefore = ProcessInfo.processInfo.thermalState.label
        let coldStart = DispatchTime.now().uptimeNanoseconds
        let predictor = try SurrogatePredictor()
        let coldOutput = try predictor.predict(numericValues: numericValues, category: category)
        let coldEnd = DispatchTime.now().uptimeNanoseconds

        var latencies = [Double]()
        var maxDrift = 0.0
        for _ in 0..<warmRuns {
            let start = DispatchTime.now().uptimeNanoseconds
            let output = try predictor.predict(numericValues: numericValues, category: category)
            let end = DispatchTime.now().uptimeNanoseconds
            latencies.append(Double(end - start) / 1_000_000)
            for (expected, actual) in zip(coldOutput, output) {
                maxDrift = max(maxDrift, abs(expected.value - actual.value))
            }
        }

        return IOSBenchmarkEvidence(
            schemaVersion: "1.0",
            capturedAtUTC: ISO8601DateFormatter().string(from: Date()),
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown",
            backend: "CoreML",
            requestedComputeUnits: "all",
            neuralEnginePlacement: "not_measured",
            powerMeasurement: "not_measured",
            thermalStateBefore: thermalBefore,
            thermalStateAfter: ProcessInfo.processInfo.thermalState.label,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            sourceModelSha256: predictor.contract.sourceModelSha256,
            preprocessingSha256: predictor.contract.preprocessingSha256,
            contractSha256: predictor.contractSHA256,
            device: IOSDeviceIdentity(
                model: hardwareIdentifier(),
                systemName: UIDevice.current.systemName,
                systemVersion: UIDevice.current.systemVersion,
                simulator: ProcessInfo.processInfo.environment["SIMULATOR_DEVICE_NAME"] != nil
            ),
            latency: LatencySummary(
                coldMs: Double(coldEnd - coldStart) / 1_000_000,
                warmMeanMs: BenchmarkStatistics.mean(latencies),
                warmP95Ms: BenchmarkStatistics.percentile95(latencies),
                warmRuns: warmRuns
            ),
            outputMaxAbsDrift: maxDrift,
            outputs: coldOutput.map { PredictionValue(name: $0.name, value: $0.value) }
        )
    }

    private static func hardwareIdentifier() -> String {
        if let simulatedModel = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
            return simulatedModel
        }
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }
}

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

extension IOSBenchmarkEvidence {
    func writeTemporaryJSON() throws -> URL {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("EdgeGenBench-iOS-evidence.json")
        try encoder.encode(self).write(to: url, options: .atomic)
        return url
    }
}
