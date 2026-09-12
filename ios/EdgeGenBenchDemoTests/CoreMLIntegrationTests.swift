import XCTest
@testable import EdgeGenBenchDemo

final class CoreMLIntegrationTests: XCTestCase {
    func testBundledModelRunsAndRejectsInvalidInputs() throws {
        let predictor = try SurrogatePredictor()
        let values = predictor.contract.featureMean
        let category = try XCTUnwrap(predictor.contract.categories.first)
        let result = try predictor.predict(numericValues: values, category: category)
        XCTAssertEqual(result.count, predictor.contract.outputDimension)
        XCTAssertTrue(result.allSatisfy { $0.value.isFinite })
        XCTAssertThrowsError(try predictor.predict(numericValues: [.nan], category: category))
        XCTAssertThrowsError(try predictor.predict(numericValues: values, category: "unknown"))
        let evidence = try IOSBenchmarkRunner.run(numericValues: values, category: category)
        XCTAssertEqual(evidence.warmLatencySamplesMs?.count, 100)
        XCTAssertLessThanOrEqual(evidence.outputMaxAbsDrift, 1e-6)
        let attachment = XCTAttachment(data: try JSONEncoder().encode(evidence), uniformTypeIdentifier: "public.json")
        attachment.name = "CoreML execution evidence"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
