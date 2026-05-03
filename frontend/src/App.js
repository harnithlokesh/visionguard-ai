import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import "./App.css";

function App() {
  const [original, setOriginal] = useState(null);
  const [edited, setEdited] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!original || !edited) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("original", original);
    formData.append("edited", edited);

    try {
      const res = await axios.post("http://127.0.0.1:5000/compare", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <motion.h1 
        className="title"
        initial={{ opacity: 0, y: -40 }}
        animate={{ opacity: 1, y: 0 }}
      >
        VisionGuard AI
      </motion.h1>

      <motion.div 
        className="card"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        <label className="upload">
          <span>Upload Original</span>
          <input type="file" onChange={(e) => setOriginal(e.target.files[0])} />
        </label>

        <label className="upload">
          <span>Upload Edited</span>
          <input type="file" onChange={(e) => setEdited(e.target.files[0])} />
        </label>

        <button onClick={handleSubmit}>
          {loading ? "Analyzing..." : "Compare"}
        </button>
      </motion.div>

      {result && (
        <motion.div 
          className="result"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h2>{result.verdict}</h2>
          <p>{result.similarity}% Similarity</p>

          <img
            src={`http://127.0.0.1:5000${result.image_path}`}
            alt="output"
          />
        </motion.div>
      )}
    </div>
  );
}

export default App;