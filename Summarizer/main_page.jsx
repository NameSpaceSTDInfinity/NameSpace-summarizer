import React, { useState } from "react";
import axios from "axios";

const Summarizer = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [summary, setSummary] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Handle file upload
  const handleFileChange = (e) => {
    setPdfFile(e.target.files[0]);
  };

  // Upload PDF and get summary
  const handleUpload = async () => {
    if (!pdfFile) {
      alert("Please upload a PDF file");
      return;
    }

    const formData = new FormData();
    formData.append("file", pdfFile);

    try {
      setIsLoading(true);
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setSummary(response.data.summary);
    } catch (error) {
      console.error("Error uploading file", error);
      alert("Error generating summary");
    } finally {
      setIsLoading(false);
    }
  };

  // Save summary to text file
  const handleSaveSummary = () => {
    const blob = new Blob([summary], { type: "text/plain;charset=utf-8" });
    const link = document.createElement("a");
    link.href = window.URL.createObjectURL(blob);
    link.download = "summary.txt";
    link.click();
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>PDF Summarizer</h2>
      
      <input type="file" accept="application/pdf" onChange={handleFileChange} />
      
      <button onClick={handleUpload} disabled={isLoading}>
        {isLoading ? "Generating summary..." : "Upload and Summarize"}
      </button>
      
      {summary && (
        <div>
          <h3>Summary:</h3>
          <textarea value={summary} readOnly rows="10" cols="80" />
          <br />
          <button onClick={handleSaveSummary}>Save Summary</button>
        </div>
      )}
    </div>
  );
};

export default Summarizer;
