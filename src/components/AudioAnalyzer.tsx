import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface AudioAnalyzerProps {
  files: File[];
}

const AudioAnalyzer: React.FC<AudioAnalyzerProps> = ({ files }) => {
  const [frequencyData, setFrequencyData] = useState<number[][]>([]);
  const [durations, setDurations] = useState<number[]>([]);
  const [matchPercentage, setMatchPercentage] = useState<number | null>(null);

  useEffect(() => {
    const analyzeAudios = async () => {
      const newFrequencyData: number[][] = [];
      const newDurations: number[] = [];

      for (const file of files) {
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const analyser = audioContext.createAnalyser();
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(analyser);
        analyser.connect(audioContext.destination);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);

        newFrequencyData.push(Array.from(dataArray));
        newDurations.push(audioBuffer.duration);

        source.disconnect();
        analyser.disconnect();
        await audioContext.close();
      }

      setFrequencyData(newFrequencyData);
      setDurations(newDurations);

      if (newFrequencyData.length === 2) {
        const similarity = 0.5;
        setMatchPercentage(similarity * 100);
      }

      console.log(
        "1st Details ===>",
        newFrequencyData[0],
        typeof newFrequencyData[0]
      );
      console.log(
        "2st Details ===>",
        newFrequencyData[1],
        typeof newFrequencyData[1]
      );
    };

    if (files.length > 0) {
      analyzeAudios();
    }
  }, [files]);

  const chartData = {
    labels: Array.from({ length: frequencyData[0]?.length || 0 }, (_, i) => i),
    datasets: frequencyData.map((data, index) => ({
      label: `Audio ${index + 1}`,
      data: data,
      borderColor: index === 0 ? "rgb(75, 192, 192)" : "rgb(255, 99, 132)",
      tension: 0.1,
    })),
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "Frequency Spectrum",
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Frequency Bin",
        },
      },
      y: {
        title: {
          display: true,
          text: "Magnitude",
        },
      },
    },
  };

  return (
    <div
      style={{
        height: "calc(100vh - 25px)",
        display: "flex",
        flexDirection: "column",
        placeItems: "center",
        justifyContent: "center",
      }}
      className="flex flex-col items-center justify-start h-full"
    >
      <h2>Audio Analysis Results</h2>
      {durations.map((duration, index) => (
        <p key={index}>
          Audio {index + 1} Duration: {duration.toFixed(2)} seconds
        </p>
      ))}
      {matchPercentage !== null && (
        <p>Match Percentage: {matchPercentage.toFixed(2)}%</p>
      )}
      {frequencyData.length > 0 && (
        <div
          style={{
            height: "calc(100vh - 255px)",
            width: "calc(100vw - 165px)",
            display: "flex",
            flexDirection: "column",
            placeItems: "center",
            justifyContent: "center",
          }}
        >
          <Line data={chartData} options={chartOptions} />
        </div>
      )}
    </div>
  );
};

export default AudioAnalyzer;
