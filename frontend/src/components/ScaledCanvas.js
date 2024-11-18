import React, { useRef, useEffect } from "react";

export const ScaledCanvas = ({ inferenceData }) => {
  const scaledCanvasRef = useRef(null);

  useEffect(() => {
    // You can update the scaled canvas based on inferenceData if needed
  }, [inferenceData]);

  return (
    <canvas
      id="scaled-canvas"
      ref={scaledCanvasRef}
      width="28"
      height="28"
      style={{ border: "1px solid #aaa", width: "100px", height: "100px" }}
    />
  );
};
