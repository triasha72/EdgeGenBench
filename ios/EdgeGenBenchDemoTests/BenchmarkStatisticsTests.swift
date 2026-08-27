import XCTest
@testable import EdgeGenBenchDemo

final class BenchmarkStatisticsTests: XCTestCase {
    func testMeanAndNearestRankP95() {
        let values = Array(1...100).map(Double.init)
        XCTAssertEqual(BenchmarkStatistics.mean(values), 50.5)
        XCTAssertEqual(BenchmarkStatistics.percentile95(values), 95.0)
    }

    func testEvidenceRoundTripsAsJSON() throws {
        let evidence = IOSBenchmarkEvidence(
            schemaVersion: "1.0",
            capturedAtUTC: "2026-08-27T00:00:00Z",
            appVersion: "0.1.0",
            backend: "CoreML",
            requestedComputeUnits: "all",
            neuralEnginePlacement: "not_measured",
            powerMeasurement: "not_measured",
            thermalStateBefore: "nominal",
            thermalStateAfter: "nominal",
            lowPowerMode: false,
            sourceModelSha256: String(repeating: "a", count: 64),
            preprocessingSha256: String(repeating: "b", count: 64),
            contractSha256: String(repeating: "c", count: 64),
            device: IOSDeviceIdentity(model: "iPhone", systemName: "iOS", systemVersion: "17.0", simulator: true),
            latency: LatencySummary(coldMs: 2, warmMeanMs: 1, warmP95Ms: 1.2, warmRuns: 100),
            outputMaxAbsDrift: 0,
            outputs: [PredictionValue(name: "mass", value: 1)]
        )
        let data = try JSONEncoder().encode(evidence)
        let decoded = try JSONDecoder().decode(IOSBenchmarkEvidence.self, from: data)
        XCTAssertEqual(decoded.latency.warmRuns, 100)
        XCTAssertEqual(decoded.backend, "CoreML")
    }
}
