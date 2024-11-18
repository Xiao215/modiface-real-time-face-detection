import React, { useState } from "react";
import "./App.css";
import { FabricCanvas } from "./components/FabricCanvas";
import { ScaledCanvas } from "./components/ScaledCanvas";
import { ProbabilityChart } from "./components/ProbabilityChart";

function App() {
  const [inferenceData, setInferenceData] = useState(null);
  const [clearCanvas, setClearCanvas] = useState(false);

  const handleClear = () => {
    setClearCanvas(true);
    setInferenceData(null);
  };

  return (
    <div className="App">
      <h1>Burn MNIST Inference Demo</h1>
      <table>
        <thead>
          <tr>
            <th>Draw a digit here</th>
            <th>Cropped and scaled</th>
            <th>Probability result</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <FabricCanvas
                onInference={setInferenceData}
                clearCanvas={clearCanvas}
                setClearCanvas={setClearCanvas}
              />
            </td>
            <td>
              <ScaledCanvas inferenceData={inferenceData} />
            </td>
            <td>
              <ProbabilityChart data={inferenceData} />
            </td>
          </tr>
          <tr>
            <td>
              <button id="clear" onClick={handleClear}>
                Clear
              </button>
            </td>
            <td></td>
            <td></td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default App;
