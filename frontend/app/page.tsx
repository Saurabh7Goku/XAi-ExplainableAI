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
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 text-white py-16 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-5xl md:text-6xl font-bold mb-6 flex items-center justify-center gap-4">
            <span className="text-6xl">🥭</span>
            Mango Leaf Disease Detector
            <span className="text-6xl">🔍</span>
          </h1>
          <p className="text-xl md:text-2xl text-green-100 max-w-3xl mx-auto">
            Advanced AI-powered analysis for mango leaf health using Vision Transformers and Explainable AI
          </p>
        </div>
      </div>

      {/* Upload Section */}
      <div className="py-12 px-6 bg-white">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">Upload Your Mango Leaf Image</h2>
            <p className="text-gray-600 text-lg">Get instant disease diagnosis with detailed explanations</p>
          </div>
          <UploadForm onUpload={handleImageUpload} loading={loading} />
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="px-6 py-8">
          <div className="max-w-4xl mx-auto">
            <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg shadow-sm">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <svg className="h-6 w-6 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-red-800">Analysis Failed</h3>
                  <p className="text-red-700 mt-1">{error}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results Section - Multi-Grid Layout */}
      {prediction && (
        <div className="py-16 px-6">
          <div className="max-w-7xl mx-auto">
            {/* Results Header */}
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-gray-800 mb-4">Analysis Results</h2>
              <div className="w-24 h-1 bg-gradient-to-r from-green-500 to-teal-500 mx-auto rounded-full"></div>
            </div>

            {/* Main Results Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
              {/* Left Column - Primary Results */}
              <div className="lg:col-span-2 space-y-8">
                <ResultsDisplay
                  prediction={prediction.prediction}
                  confidence={prediction.confidence}
                  limeExplanation={prediction.lime_explanation}
                  classProbabilities={prediction.class_probabilities}
                />
              </div>

              {/* Right Column - Disease Info & Stats */}
              <div className="space-y-8">
                {/* Quick Stats Card */}
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-6 border border-blue-100">
                  <h3 className="text-xl font-bold text-blue-900 mb-4 flex items-center gap-2">
                    <span className="text-2xl">📊</span>
                    Analysis Summary
                  </h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-blue-700 font-medium">Confidence</span>
                      <span className="text-blue-900 font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-blue-700 font-medium">Disease</span>
                      <span className="text-blue-900 font-bold">{prediction.prediction}</span>
                    </div>
                  </div>
                </div>

                {/* Disease Information Card */}
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-6 border border-green-100">
                  <h3 className="text-xl font-bold text-green-900 mb-4 flex items-center gap-2">
                    <span className="text-2xl">🌿</span>
                    Disease Information
                  </h3>
                  <div className="space-y-3">
                    <div>
                      <h4 className="font-semibold text-green-800 mb-1">Symptoms</h4>
                      <ul className="text-sm text-green-700 space-y-1">
                        {prediction.disease_info.symptoms?.slice(0, 3).map((symptom: string, idx: number) => (
                          <li key={idx} className="flex items-start gap-2">
                            <span className="text-green-500 mt-1">•</span>
                            {symptom}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold text-green-800 mb-1">Treatments</h4>
                      <ul className="text-sm text-green-700 space-y-1">
                        {prediction.disease_info.treatments?.slice(0, 3).map((treatment: string, idx: number) => (
                          <li key={idx} className="flex items-start gap-2">
                            <span className="text-green-500 mt-1">•</span>
                            {treatment}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Full Width Report Section */}
            <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
              <div className="bg-gradient-to-r from-green-600 to-teal-600 px-8 py-6">
                <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                  <span className="text-3xl">📋</span>
                  Detailed Expert Report
                </h3>
              </div>
              <div className="p-8">
                <ReportCard
                  diseaseName={prediction.prediction}
                  symptoms={prediction.disease_info.symptoms}
                  treatments={prediction.disease_info.treatments}
                  report={prediction.report}
                  confidence={prediction.confidence}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12 px-6 mt-16">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <span className="text-2xl">🥭</span>
                Mango AI
              </h3>
              <p className="text-gray-300">
                Advanced agricultural technology for sustainable farming practices.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Technology</h4>
              <ul className="space-y-2 text-gray-300">
                <li>PyTorch Vision Transformer</li>
                <li>LIME Explainable AI</li>
                <li>Gemini LLM Integration</li>
                <li>Next.js Frontend</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Features</h4>
              <ul className="space-y-2 text-gray-300">
                <li>Real-time Disease Detection</li>
                <li>Visual Explanations</li>
                <li>Expert Recommendations</li>
                <li>Confidence Scoring</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-700 mt-8 pt-8 text-center">
            <p className="text-gray-400">Built with ❤️ for farmers worldwide • Powered by XAI</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
