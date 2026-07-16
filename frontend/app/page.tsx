'use client'

import { useState, useRef } from 'react'
import UploadForm from './components/UploadForm'
import ResultsDisplay from './components/ResultsDisplay'
import ReportCard from './components/ReportCard'
import { usePrediction } from './hooks/useApi'

export default function Home() {
  const { prediction, loading, error, predict, reset } = usePrediction()
  const [showScrollHint, setShowScrollHint] = useState(false)
  const resultsRef = useRef<HTMLDivElement>(null)

  const handleImageUpload = async (file: File) => {
    reset()
    try {
      await predict(file)
      setShowScrollHint(true)
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 300)
    } catch (err) {
      console.error('Prediction error:', err)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 bg-grid-pattern">
      {/* Floating Background Decorations */}
      <div className="fixed top-20 left-10 text-6xl opacity-10 animate-float pointer-events-none" style={{ animationDelay: '0s' }}>🥭</div>
      <div className="fixed top-40 right-16 text-5xl opacity-10 animate-float pointer-events-none" style={{ animationDelay: '1s' }}>🍃</div>
      <div className="fixed bottom-40 left-20 text-6xl opacity-10 animate-float pointer-events-none" style={{ animationDelay: '2s' }}>🌿</div>
      <div className="fixed bottom-20 right-10 text-5xl opacity-10 animate-float pointer-events-none" style={{ animationDelay: '0.5s' }}>🌱</div>

      {/* Navigation Bar */}
      <nav className="relative z-10 bg-white/80 backdrop-blur-md border-b border-green-100 sticky top-0">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🥭</span>
            <span className="text-xl font-bold bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent">
              MangoAI
            </span>
          </div>
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <span className="hidden sm:inline">Powered by Vision Transformer</span>
            <span className="hidden sm:inline">|</span>
            <span className="hidden sm:inline">XAI-Driven</span>
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <span className="text-green-600 font-medium">Live</span>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Gradient Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-green-600 via-emerald-600 to-teal-700"></div>
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width%3D%2260%22 height%3D%2260%22 viewBox%3D%220 0 60 60%22 xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg fill%3D%22none%22 fill-rule%3D%22evenodd%22%3E%3Cg fill%3D%22%23ffffff%22 fill-opacity%3D%220.05%22%3E%3Cpath d%3D%22M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z%22%2F%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E')] opacity-40"></div>

        <div className="relative max-w-7xl mx-auto px-2 py-10 md:py-28 text-center">
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold text-white mb-6 leading-tight">
            <span className="inline-block animate-fade-in-up">Mango</span>{' '}
            <span className="inline-block animate-fade-in-up" style={{ animationDelay: '0.1s' }}>Leaf</span>{' '}
            <span className="inline-block bg-gradient-to-r from-yellow-300 to-orange-300 bg-clip-text text-transparent animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
              Disease
            </span>{' '}
            <span className="inline-block animate-fade-in-up" style={{ animationDelay: '0.3s' }}>Detector</span>
          </h1>

          <p className="text-xl md:text-2xl text-green-100 max-w-4xl mx-auto mb-12 animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
            Advanced AI-powered analysis for mango leaf health using <strong className="text-white">Vision Transformers</strong> and <strong className="text-white">Explainable AI</strong>
          </p>
          <p className="text-base text-green-200/80 max-w-3xl mx-auto animate-fade-in-up" style={{ animationDelay: '0.5s' }}>
            <span className="font-medium text-green-100">Focus:</span> Demonstrating transparent AI reasoning — visualizing which leaf features drive each prediction
          </p>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto mb-16 animate-fade-in-up" style={{ animationDelay: '0.5s' }}>
            {[
              { icon: '🧠', title: 'Deep Learning', desc: 'ViT-based neural network trained on 10,000+ mango leaf images' },
              { icon: '🔍', title: 'XAI Transparency', desc: 'Attention rollout shows exactly which leaf regions triggered the diagnosis' },
              { icon: '📋', title: 'Smart Reports', desc: 'Get detailed actionable recommendations for disease management' },
            ].map((feature, i) => (
              <div key={i} className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/10 hover:bg-white/15 transition-all hover:scale-105">
                <div className="text-4xl mb-3">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-green-200 text-sm">{feature.desc}</p>
              </div>
            ))}
          </div>

          {/* Scroll Indicator */}
          <div className="animate-bounce mt-4">
            <svg className="w-6 h-6 text-white/60 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        </div>
      </section>

      {/* Upload Section - Full Width */}
      <section className="py-16 px-6 relative bg-gradient-to-b from-white to-green-50/30">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12 animate-fade-in-up">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-green-100 text-green-700 text-sm font-medium mb-4">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              Try It Now
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
              Analyze Your Leaf
            </h2>
            <div className="w-24 h-1 bg-gradient-to-r from-green-500 to-teal-500 mx-auto rounded-full mb-6"></div>
            <p className="text-gray-600 text-lg max-w-3xl mx-auto">
              Upload a clear photo of the mango leaf you want to analyze, or click any sample image below to instantly test the AI.
            </p>
          </div>

          {/* Two Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left — Sample Images (8/12) */}
            <div className="lg:col-span-8 animate-slide-in-left">
              <div className="bg-white rounded-3xl shadow-xl shadow-green-200/30 p-6 md:p-8 border border-green-100">
                <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
                  <div>
                    <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                      <span>📸</span>
                      Sample Images
                    </h3>
                    <p className="text-xs text-gray-400 mt-1">Click any image to test the AI — all 8 disease categories covered</p>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-gray-400 bg-gray-50 px-3 py-1.5 rounded-full">
                    <span className="w-2 h-2 rounded-full bg-green-500"></span>
                    <span>24 samples</span>
                    <span className="text-gray-300">|</span>
                    <span>8 classes</span>
                    <span className="text-gray-300">|</span>
                    <span>3 each</span>
                  </div>
                </div>

                {/* Category Tabs */}
                <div className="flex flex-wrap gap-2 mb-5">
                  {['Healthy', 'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould'].map((cat) => (
                    <span key={cat} className="px-3 py-1 bg-green-50 text-green-700 text-xs font-medium rounded-full border border-green-200">
                      {cat}
                    </span>
                  ))}
                </div>

                {/* 4-column grid for samples */}
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                  {[
                    { src: '/samples/Healthy-1.jpg', cat: 'Healthy' },
                    { src: '/samples/Healthy-2.jpg', cat: 'Healthy' },
                    { src: '/samples/Healthy-3.jpg', cat: 'Healthy' },
                    { src: '/samples/Anthracnose-1.jpg', cat: 'Anthracnose' },
                    { src: '/samples/Anthracnose-2.jpg', cat: 'Anthracnose' },
                    { src: '/samples/Anthracnose-3.jpg', cat: 'Anthracnose' },
                    { src: '/samples/Bacterial-Canker-1.jpg', cat: 'Bacterial Canker' },
                    { src: '/samples/Bacterial-Canker-2.jpg', cat: 'Bacterial Canker' },
                    { src: '/samples/Bacterial-Canker-3.jpg', cat: 'Bacterial Canker' },
                    { src: '/samples/Cutting-Weevil-1.jpg', cat: 'Cutting Weevil' },
                    { src: '/samples/Cutting-Weevil-2.jpg', cat: 'Cutting Weevil' },
                    { src: '/samples/Cutting-Weevil-3.jpg', cat: 'Cutting Weevil' },
                    { src: '/samples/Die-Back-1.jpg', cat: 'Die Back' },
                    { src: '/samples/Die-Back-2.jpg', cat: 'Die Back' },
                    { src: '/samples/Die-Back-3.jpg', cat: 'Die Back' },
                    { src: '/samples/Gall-Midge-1.jpg', cat: 'Gall Midge' },
                    { src: '/samples/Gall-Midge-2.jpg', cat: 'Gall Midge' },
                    { src: '/samples/Gall-Midge-3.jpg', cat: 'Gall Midge' },
                    { src: '/samples/Powdery-Mildew-1.jpg', cat: 'Powdery Mildew' },
                    { src: '/samples/Powdery-Mildew-2.jpg', cat: 'Powdery Mildew' },
                    { src: '/samples/Powdery-Mildew-3.jpg', cat: 'Powdery Mildew' },
                    { src: '/samples/Sooty-Mould-1.jpg', cat: 'Sooty Mould' },
                    { src: '/samples/Sooty-Mould-2.jpg', cat: 'Sooty Mould' },
                    { src: '/samples/Sooty-Mould-3.jpg', cat: 'Sooty Mould' },
                  ].map((sample, idx) => {
                    // Color coding per category
                    const colorMap: Record<string, string> = {
                      'Healthy': 'border-green-300 hover:border-green-500',
                      'Anthracnose': 'border-amber-300 hover:border-amber-500',
                      'Bacterial Canker': 'border-red-300 hover:border-red-500',
                      'Cutting Weevil': 'border-orange-300 hover:border-orange-500',
                      'Die Back': 'border-rose-300 hover:border-rose-500',
                      'Gall Midge': 'border-purple-300 hover:border-purple-500',
                      'Powdery Mildew': 'border-yellow-300 hover:border-yellow-500',
                      'Sooty Mould': 'border-gray-400 hover:border-gray-600',
                    }
                    return (
                      <button
                        key={idx}
                        onClick={() => {
                          fetch(sample.src)
                            .then(res => res.blob())
                            .then(blob => {
                              const file = new File([blob], `${sample.cat}.jpg`, { type: 'image/jpeg' })
                              handleImageUpload(file)
                            })
                        }}
                        disabled={loading}
                        className={`group relative rounded-xl overflow-hidden border-2 ${colorMap[sample.cat] || 'border-gray-200 hover:border-green-400'} transition-all hover:shadow-lg hover:-translate-y-1.5 ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                        title={sample.cat}
                      >
                        <div className="aspect-[4/3] bg-gray-100">
                          <img
                            src={sample.src}
                            alt={sample.cat}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                            loading="lazy"
                          />
                        </div>
                        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 via-black/30 to-transparent p-2 pt-6">
                          <span className="text-white text-[11px] font-semibold block truncate">{sample.cat}</span>
                        </div>
                        <div className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                          <span className="px-2 py-0.5 bg-green-500 text-white text-[10px] rounded-full font-bold shadow-lg">▶ Test</span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Right — Upload (4/12) */}
            <div className="lg:col-span-4 animate-slide-in-right">
              <div className="bg-white rounded-3xl shadow-xl shadow-green-200/30 p-6 md:p-8 border border-green-100 relative overflow-hidden sticky top-24">
                <div className="absolute top-0 right-0 w-48 h-48 bg-gradient-to-bl from-green-100/50 to-transparent rounded-full -translate-y-1/2 translate-x-1/2"></div>
                <div className="absolute bottom-0 left-0 w-32 h-32 bg-gradient-to-tr from-emerald-50/50 to-transparent rounded-full translate-y-1/2 -translate-x-1/2"></div>

                <div className="relative z-10">
                  <div className="text-center mb-6">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-green-100 to-emerald-100 mb-4">
                      <span className="text-3xl">📤</span>
                    </div>
                    <h3 className="text-xl font-bold text-gray-800">Upload Your Image</h3>
                    <p className="text-xs text-gray-400 mt-1">or drag & drop your own leaf photo</p>
                  </div>
                  <UploadForm onUpload={handleImageUpload} loading={loading} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Error Display */}
      {error && (
        <section className="px-6 py-8 animate-fade-in-up">
          <div className="max-w-4xl mx-auto">
            <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-xl shadow-lg">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-red-800 mb-1">Analysis Failed</h3>
                  <p className="text-red-600">{error}</p>
                </div>
                <button onClick={reset} className="text-red-400 hover:text-red-600 transition-colors">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Results Section */}
      <div ref={resultsRef}>
        {prediction && (
          <section className="py-16 px-6">
            <div className="max-w-7xl mx-auto">
              {/* Results Header */}
              <div className="text-center mb-12 animate-fade-in-up">
                <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-green-100 text-green-700 text-sm font-medium mb-4">
                  <span className="w-2 h-2 rounded-full bg-green-500"></span>
                  Analysis Complete
                </div>
                <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
                  Diagnostic Results
                </h2>
                <div className="w-24 h-1 bg-gradient-to-r from-green-500 to-teal-500 mx-auto rounded-full"></div>
              </div>

              {/* XAI Purpose Disclaimer */}
              <div className="mb-8 bg-blue-50 border-l-4 border-blue-400 rounded-lg p-6 shadow-sm">
                <div className="flex items-start gap-4">
                  <div className="text-3xl flex-shrink-0">ℹ️</div>
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-blue-900 mb-2">
                      About This Analysis
                    </h3>
                    <p className="text-blue-800 text-sm leading-relaxed mb-2">
                      This demonstration showcases <strong>Explainable AI (XAI)</strong> capabilities — the primary goal is to illuminate <strong>how the model reasons</strong> and <strong>which visual features drive the diagnosis</strong>, not just to provide a classification label.
                    </p>
                    <p className="text-blue-700 text-xs leading-relaxed">
                      The Vision Transformer model analyzes attention patterns to create transparent explanations. While the model is trained on extensive agricultural data, clinical verification by agricultural experts is recommended for definitive diagnosis.
                    </p>
                  </div>
                </div>
              </div>

              {/* Main Results Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
                {/* Left Column - Primary Results */}
                <div className="lg:col-span-2 space-y-8 animate-slide-in-left">
                  <ResultsDisplay
                    prediction={prediction.prediction}
                    confidence={prediction.confidence}
                    limeExplanation={prediction.lime_explanation}
                    classProbabilities={prediction.class_probabilities}
                  />
                </div>

                {/* Right Column - Stats & Distribution */}
                <div className="space-y-8 animate-slide-in-right">
                  {/* Class Probability Distribution Card */}
                  {prediction.class_probabilities && Object.keys(prediction.class_probabilities).length > 0 && (
                    <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6">
                      <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <span className="text-xl">📊</span>
                        Class Probability Distribution
                      </h3>
                      <div className="space-y-2.5">
                        {Object.entries(prediction.class_probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([disease, prob]) => {
                            const isTop = disease === prediction.prediction
                            const barWidth = prob * 100
                            return (
                              <div key={disease} className="group">
                                <div className="flex justify-between text-sm mb-1">
                                  <span className={`${isTop ? 'font-bold text-gray-900' : 'text-gray-600'}`}>
                                    {disease}
                                    {isTop && <span className="ml-2 text-xs text-green-500">← Predicted</span>}
                                  </span>
                                  <span className={`${isTop ? 'font-bold text-green-700' : 'text-gray-500'}`}>
                                    {(barWidth).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                                  <div
                                    className={`h-full rounded-full transition-all duration-700 ${isTop
                                      ? 'bg-gradient-to-r from-green-400 to-emerald-500'
                                      : 'bg-gray-300 group-hover:bg-gray-400'
                                      }`}
                                    style={{ width: `${barWidth}%` }}
                                  ></div>
                                </div>
                              </div>
                            )
                          })}
                      </div>
                    </div>
                  )}

                  {/* Disease Info Card */}
                  <div className="bg-white rounded-2xl p-6 border border-green-100 shadow-lg">
                    <h3 className="text-lg font-bold text-green-900 mb-4 flex items-center gap-2">
                      <span className="text-2xl">🌿</span>
                      Key Indicators
                    </h3>
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-semibold text-red-700 mb-2 flex items-center gap-1.5">
                          <span>⚠️</span> Symptoms
                        </h4>
                        <ul className="text-sm text-gray-700 space-y-1.5">
                          {prediction.disease_info.symptoms?.slice(0, 4).map((symptom: string, idx: number) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-red-500 mt-1 text-xs">●</span>
                              {symptom}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="border-t border-green-100 pt-4">
                        <h4 className="font-semibold text-blue-700 mb-2 flex items-center gap-1.5">
                          <span>💊</span> Treatments
                        </h4>
                        <ul className="text-sm text-gray-700 space-y-1.5">
                          {prediction.disease_info.treatments?.slice(0, 3).map((treatment: string, idx: number) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-blue-500 mt-1">✓</span>
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
              <div className="animate-fade-in-up">
                <div className="bg-white rounded-3xl shadow-2xl border border-gray-100 overflow-hidden">
                  <div className="relative overflow-hidden">
                    <div className="bg-gradient-to-r from-green-600 via-emerald-600 to-teal-600 px-8 py-8 relative">
                      <div className="absolute top-0 right-0 w-48 h-48 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/4"></div>
                      <div className="absolute bottom-0 left-0 w-32 h-32 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/4"></div>
                      <h3 className="text-2xl md:text-3xl font-bold text-white flex items-center gap-3 relative z-10">
                        <span className="text-3xl">📋</span>
                        Detailed Expert Analysis Report
                      </h3>
                      <p className="text-green-200 mt-2 text-sm relative z-10">
                        AI-Generated comprehensive diagnosis with treatment recommendations
                      </p>
                    </div>
                  </div>
                  <div className="p-8 md:p-10">
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

              {/* New Analysis Button */}
              <div className="text-center mt-12 animate-fade-in-up">
                <button
                  onClick={reset}
                  className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-full font-semibold text-lg shadow-xl shadow-green-500/30 hover:shadow-green-500/50 hover:scale-105 transition-all"
                >
                  <span>🔄</span>
                  Analyze Another Leaf
                </button>
              </div>
            </div>
          </section>
        )}
      </div>

      {/* Stats Section */}
      <section className="py-16 px-6 bg-white border-t border-green-100">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-800 mb-4">Why Choose MangoAI?</h2>
            <div className="w-24 h-1 bg-gradient-to-r from-green-500 to-teal-500 mx-auto rounded-full"></div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { icon: '🧠', value: '98%+', label: 'Model Performance', color: 'from-blue-500 to-cyan-500' },
              { icon: '🔬', value: '8', label: 'Detectable Diseases', color: 'from-green-500 to-emerald-500' },
              { icon: '⚡', value: '<2s', label: 'Processing Time', color: 'from-orange-500 to-red-500' },
              { icon: '🌍', value: '10K+', label: 'Training Images', color: 'from-purple-500 to-pink-500' },
            ].map((stat, i) => (
              <div key={i} className="text-center group">
                <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br ${stat.color} flex items-center justify-center text-2xl group-hover:scale-110 transition-transform shadow-lg`}>
                  {stat.icon}
                </div>
                <div className="text-3xl font-bold text-gray-800 mb-1">{stat.value}</div>
                <div className="text-gray-500 text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white py-16 px-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500"></div>
        <div className="absolute -top-20 -right-20 w-64 h-64 bg-green-500/5 rounded-full"></div>
        <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-teal-500/5 rounded-full"></div>

        <div className="max-w-7xl mx-auto relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-10">
            <div className="md:col-span-2">
              <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <span className="text-3xl">🥭</span>
                MangoAI
              </h3>
              <p className="text-gray-400 max-w-md leading-relaxed">
                Empowering farmers with cutting-edge AI technology for early disease detection in mango crops. Our Vision Transformer model provides fast, accurate, and explainable diagnoses.
              </p>
              <div className="flex gap-4 mt-6">
                {['🧬', '🤖', '📊', '🌱'].map((emoji, i) => (
                  <div key={i} className="w-10 h-10 bg-white/5 rounded-lg flex items-center justify-center hover:bg-white/10 transition-colors cursor-pointer">
                    {emoji}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">Technology Stack</h4>
              <ul className="space-y-2.5 text-gray-400">
                <li className="flex items-center gap-2"><span className="w-1 h-1 rounded-full bg-green-500"></span>PyTorch ViT</li>
                <li className="flex items-center gap-2"><span className="w-1 h-1 rounded-full bg-green-500"></span>XAI Attention Rollout</li>
                <li className="flex items-center gap-2"><span className="w-1 h-1 rounded-full bg-green-500"></span>FastAPI Backend</li>
                <li className="flex items-center gap-2"><span className="w-1 h-1 rounded-full bg-green-500"></span>Next.js Frontend</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">Key Features</h4>
              <ul className="space-y-2.5 text-gray-400">
                <li className="flex items-center gap-2"><span className="w-1 h-1 rounded-full bg-teal-500"></span>Real-time Detection</li>
                <li className="flex items-center gap-2"><span className="w-1 h-1 rounded-full bg-teal-500"></span>Visual Explanations</li>
                <li className="flex items-center gap-2"><span className="w-1 h-1 rounded-full bg-teal-500"></span>Expert Recommendations</li>
                <li className="flex items-center gap-2"><span className="w-1 h-1 rounded-full bg-teal-500"></span>Confidence Scoring</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-700/50 mt-10 pt-8 text-center">
            <p className="text-gray-500 text-sm">
              Built with ❤️ for farmers worldwide • Powered by Explainable AI Technology
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}