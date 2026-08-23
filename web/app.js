const defaults = [65, 950, 535, 525, 0.57, 0.24];
const labels = {
  passenger_capacity: "Passenger capacity",
  design_range_km: "Design range (km)",
  cruise_speed_kmh: "Cruise speed (km/h)",
  battery_specific_energy_wh_per_kg: "Battery specific energy (Wh/kg)",
  hydrogen_storage_efficiency: "Hydrogen storage efficiency",
  hybridization_ratio: "Hybridization ratio",
  estimated_takeoff_mass_kg: "Takeoff mass (kg)",
  mission_energy_kwh: "Mission energy (kWh)",
  energy_per_passenger_km_kwh: "Energy / passenger-km (kWh)",
  lifecycle_emissions_proxy_kgco2e: "Lifecycle emissions proxy (kg CO₂e)",
  operating_cost_proxy_usd: "Operating cost proxy (USD)",
  noise_proxy_db: "Noise proxy (dB)",
};

let contract;
let session;

const status = document.querySelector("#status");
const runButton = document.querySelector("#run");

async function initialise() {
  try {
    contract = await fetch("./model-contract.json").then(checkResponse).then((response) => response.json());
    buildForm(contract);
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    session = await ort.InferenceSession.create("./model/neural_surrogate.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    runButton.disabled = false;
    runButton.textContent = "Run on this device";
    status.textContent = "Model ready. Inference stays on this device.";
  } catch (error) {
    status.textContent = `The local model could not load: ${error.message}`;
  }
}

function checkResponse(response) {
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response;
}

function buildForm(modelContract) {
  const fields = document.querySelector("#fields");
  modelContract.numericFeatures.forEach((name, index) => {
    const label = document.createElement("label");
    label.textContent = labels[name] || name;
    const input = document.createElement("input");
    input.type = "number";
    input.name = name;
    input.required = true;
    input.step = "any";
    input.value = defaults[index];
    label.append(input);
    fields.append(label);
  });
  const category = document.querySelector("#category");
  modelContract.categories.forEach((name) => category.add(new Option(name.replaceAll("_", " "), name)));
}

document.querySelector("#design-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!session) return;
  runButton.disabled = true;
  status.textContent = "Running the surrogate…";
  try {
    const form = new FormData(event.currentTarget);
    const numeric = contract.numericFeatures.map((name) => Number(form.get(name)));
    if (numeric.some((value) => !Number.isFinite(value))) throw new Error("Enter a number in every field.");
    const encoded = numeric.map((value, index) => (value - contract.featureMean[index]) / contract.featureScale[index]);
    const selected = form.get(contract.categoricalFeature);
    contract.categories.forEach((name) => encoded.push(name === selected ? 1 : 0));
    const tensor = new ort.Tensor("float32", Float32Array.from(encoded), [1, contract.inputDimension]);
    const output = await session.run({ [contract.inputName]: tensor });
    const normalized = output[contract.outputName].data;
    const predictions = contract.targets.map((name, index) => ({
      name,
      value: normalized[index] * contract.targetScale[index] + contract.targetMean[index],
    }));
    renderResults(predictions);
    status.textContent = "Inference completed locally with ONNX Runtime Web.";
  } catch (error) {
    status.textContent = `Inference failed: ${error.message}`;
  } finally {
    runButton.disabled = false;
  }
});

function renderResults(predictions) {
  const results = document.querySelector("#results");
  results.replaceChildren(...predictions.map(({ name, value }) => {
    const row = document.createElement("div");
    row.className = "result";
    const title = document.createElement("span");
    title.textContent = labels[name] || name;
    const number = document.createElement("strong");
    number.textContent = new Intl.NumberFormat(undefined, { maximumFractionDigits: name.includes("passenger_km") ? 4 : 2 }).format(value);
    row.append(title, number);
    return row;
  }));
}

if ("serviceWorker" in navigator) window.addEventListener("load", () => navigator.serviceWorker.register("./service-worker.js"));
initialise();
