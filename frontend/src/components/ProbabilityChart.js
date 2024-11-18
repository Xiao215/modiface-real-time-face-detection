import React, { useEffect, useRef } from "react";
import { Chart } from "chart.js";
import { chartConfigBuilder } from "../utils";

export const ProbabilityChart = ({ data }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    chartInstance.current = chartConfigBuilder(chartRef.current);
  }, []);

  useEffect(() => {
    if (chartInstance.current && data) {
      chartInstance.current.data.datasets[0].data = data;
      chartInstance.current.update();
    }
  }, [data]);

  return (
    <canvas
      id="chart"
      ref={chartRef}
      style={{ width: "600px", height: "300px" }}
    />
  );
};
