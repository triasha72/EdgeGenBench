import CoreML
import CryptoKit
import Foundation

struct ModelContract: Decodable {
    let schemaVersion: String
    let sourceModelSha256: String
    let preprocessingSha256: String
    let inputName: String
    let outputName: String
    let numericFeatures: [String]
    let categoricalFeature: String
    let categories: [String]
    let featureMean: [Double]
    let featureScale: [Double]
    let targets: [String]
    let targetMean: [Double]
    let targetScale: [Double]
    let inputDimension: Int
    let outputDimension: Int
}

struct Prediction: Identifiable {
    let name: String
    let value: Double
    var id: String { name }
}

enum SurrogateError: LocalizedError {
    case missingResource(String)
    case invalidContract(String)
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .missingResource(let name): return "Missing bundled resource: \(name)"
        case .invalidContract(let detail): return "Invalid model contract: \(detail)"
        case .invalidOutput: return "Core ML returned an unexpected output tensor."
        }
    }
}

final class SurrogatePredictor {
    let contract: ModelContract
    let contractSHA256: String
    private let model: MLModel

    init(bundle: Bundle = .main) throws {
        guard let contractURL = bundle.url(forResource: "ModelContract", withExtension: "json") else {
            throw SurrogateError.missingResource("ModelContract.json")
        }
        let contractData = try Data(contentsOf: contractURL)
        contract = try JSONDecoder().decode(ModelContract.self, from: contractData)
        contractSHA256 = SHA256.hash(data: contractData).map { String(format: "%02x", $0) }.joined()
        try contract.validate()
        guard contract.featureMean.count + contract.categories.count == contract.inputDimension,
              contract.featureScale.count == contract.featureMean.count,
              contract.targets.count == contract.outputDimension,
              contract.targetMean.count == contract.outputDimension,
              contract.targetScale.count == contract.outputDimension else {
            throw SurrogateError.invalidContract("array dimensions do not agree")
        }
        guard let modelURL = bundle.url(forResource: "NeuralSurrogate", withExtension: "mlmodelc") else {
            throw SurrogateError.missingResource("NeuralSurrogate.mlpackage")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
        guard model.modelDescription.inputDescriptionsByName[contract.inputName] != nil,
              model.modelDescription.outputDescriptionsByName[contract.outputName] != nil else {
            throw SurrogateError.invalidContract("Core ML feature names do not agree")
        }
    }

    func predict(numericValues: [Double], category: String) throws -> [Prediction] {
        guard numericValues.allSatisfy({ $0.isFinite }),
              numericValues.count == contract.featureMean.count,
              let categoryIndex = contract.categories.firstIndex(of: category) else {
            throw SurrogateError.invalidContract("input values do not agree")
        }
        let features = try MLMultiArray(shape: [1, NSNumber(value: contract.inputDimension)], dataType: .float32)
        for index in numericValues.indices {
            features[index] = NSNumber(value: (numericValues[index] - contract.featureMean[index]) / contract.featureScale[index])
        }
        for index in contract.categories.indices {
            features[numericValues.count + index] = NSNumber(value: index == categoryIndex ? 1.0 : 0.0)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [contract.inputName: features])
        let result = try model.prediction(from: provider)
        guard let normalized = result.featureValue(for: contract.outputName)?.multiArrayValue,
              normalized.count == contract.outputDimension else {
            throw SurrogateError.invalidOutput
        }
        guard (0..<normalized.count).allSatisfy({ normalized[$0].doubleValue.isFinite }) else {
            throw SurrogateError.invalidOutput
        }
        let predictions = contract.targets.indices.map { index in
            Prediction(name: contract.targets[index], value: normalized[index].doubleValue * contract.targetScale[index] + contract.targetMean[index])
        }
        guard predictions.allSatisfy({ $0.value.isFinite }) else {
            throw SurrogateError.invalidOutput
        }
        return predictions
    }
}


extension ModelContract {
    func validate() throws {
        guard schemaVersion == "1.0", !numericFeatures.isEmpty,
              numericFeatures.count == featureMean.count,
              featureMean.count == featureScale.count,
              Set(numericFeatures).count == numericFeatures.count,
              !categories.isEmpty, Set(categories).count == categories.count,
              featureMean.allSatisfy({ $0.isFinite }),
              featureScale.allSatisfy({ $0.isFinite && $0 > 0 }),
              targetMean.allSatisfy({ $0.isFinite }),
              targetScale.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw SurrogateError.invalidContract("nonfinite, duplicate, or invalid preprocessing values")
        }
    }
}
