'use client'

import { useState } from 'react'
import UploadForm from './components/UploadForm'
import ResultsDisplay from './components/ResultsDisplay'
import ReportCard from './components/ReportCard'
import { usePrediction } from './hooks/useApi'

export default function Home() {
  const { prediction, loading, error, predict, reset } = usePrediction()

  const handleImageUpload = async (file: File) => {
    reset()
    try {
      await predict(file)
    } catch (err) {
      // Error is already handled by the hook
      console.error('Prediction error:', err)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-green-800 mb-4">
            🥭 Mango Leaf Disease Detector
          </h1>
        </header>

        <main className="bg-white rounded-2xl shadow-xl p-6 md:p-8">
          <UploadForm onUpload={handleImageUpload} loading={loading} />

          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              Error: {error}
            </div>
          )}

          {prediction && (
            <div className="mt-8 space-y-8">
              <ResultsDisplay
                prediction={prediction.prediction}
                confidence={prediction.confidence}
                limeExplanation={prediction.lime_explanation}
                classProbabilities={prediction.class_probabilities}
              />

              <ReportCard
                diseaseName={prediction.prediction}
                symptoms={prediction.disease_info.symptoms}
                treatments={prediction.disease_info.treatments}
                report={prediction.report}
                confidence={prediction.confidence}
              />
            </div>
          )}
        </main>

        <footer className="mt-12 text-center text-gray-600 text-sm">
          <p>Built with PyTorch Vision Transformer • Powered by XAi</p>
        </footer>
      </div>
    </div>
  )
}
