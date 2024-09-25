import React, { useState } from "react";
import AudioAnalyzer from "./AudioAnalyzer";

const AudioUploader: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFiles(Array.from(event.target.files).slice(0, 2));
    }
  };

  return (
    <div>
      <input
        type="file"
        accept="audio/*"
        multiple
        onChange={handleFileChange}
      />
      {files.length > 0 && <AudioAnalyzer files={files} />}
    </div>
  );
};

export default AudioUploader;
