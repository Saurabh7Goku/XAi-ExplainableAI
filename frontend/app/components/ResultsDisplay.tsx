'use client'

interface ResultsDisplayProps {
  prediction: string
  confidence: number
  limeExplanation: string
  classProbabilities?: Record<string, number>
}

export default function ResultsDisplay({
  prediction,
  confidence,
  limeExplanation,
  classProbabilities
}: ResultsDisplayProps) {
  const confidencePercentage = Math.round(confidence * 100)
  const isHighConfidence = confidence > 0.7
  const isHealthy = prediction === 'Healthy'

  return (
    <div className="space-y-6">
      {/* Combined Row: Diagnosis Result + Detected Condition */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Diagnosis Result Card (compact) */}
        <div className={`rounded-2xl p-4 shadow-lg border-2 relative overflow-hidden ${isHealthy
          ? 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-200'
          : 'bg-gradient-to-br from-red-50 to-orange-50 border-red-200'
          }`}>
          <div className="relative z-10">
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg shadow-lg ${isHealthy
                ? 'bg-gradient-to-br from-green-400 to-emerald-500'
                : 'bg-gradient-to-br from-red-400 to-orange-500'
                }`}>
                {isHealthy ? '✅' : '⚠️'}
              </div>
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Diagnosis</p>
                <h2 className={`text-lg md:text-xl font-bold ${isHealthy ? 'text-green-800' : 'text-red-800'}`}>
                  {prediction}
                </h2>
              </div>
            </div>
            {/* Confidence Bar */}
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs font-medium text-gray-500">Confidence</span>
                <span className={`text-xs font-bold ${isHighConfidence ? 'text-green-600' : 'text-yellow-600'}`}>
                  {confidencePercentage}%
                </span>
              </div>
              <div className="h-2 bg-gray-200/70 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ease-out ${isHighConfidence
                    ? 'bg-gradient-to-r from-green-400 to-emerald-500'
                    : 'bg-gradient-to-r from-yellow-400 to-orange-500'
                    }`}
                  style={{ width: `${confidencePercentage}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Detected Condition Card (compact) */}
        <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-2xl p-4 border border-amber-200 shadow-lg flex items-center">
          <div className="w-full text-center">
            <div className="text-3xl mb-1">🦠</div>
            <h4 className="text-lg font-bold text-gray-800 mb-1">{prediction}</h4>
            <div className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-700 text-xs font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
              {isHealthy ? 'No treatment needed' : 'Treatment recommended'}
            </div>
          </div>
        </div>
      </div>

      {/* XAI Heatmap Card (slightly larger) */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
        <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 px-5 py-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <span className="text-lg">🔍</span>
            XAI Attention Heatmap
          </h3>
        </div>
        <div className="p-4">
          <div className="rounded-lg overflow-hidden bg-gray-50 min-h-[280px] max-h-[400px] flex items-center justify-center relative border border-gray-200">
            {limeExplanation ? (
              <div className="w-full h-full flex items-center justify-center animate-scale-in">
                <img
                  src={`data:image/png;base64,${limeExplanation}`}
                  alt="Attention heatmap overlay"
                  className="w-full h-auto max-h-[370px] object-contain"
                />
                <div className="absolute bottom-3 left-3 right-3 flex justify-center gap-3">
                  <span className="px-2.5 py-1 bg-black/60 text-white text-xs rounded-full backdrop-blur-sm">
                    🔴 High attention
                  </span>
                  <span className="px-2.5 py-1 bg-black/60 text-white text-xs rounded-full backdrop-blur-sm">
                    🔵 Low attention
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center p-8">
                <div className="relative w-20 h-20 mx-auto mb-6">
                  <div className="absolute inset-0 rounded-full border-4 border-purple-100"></div>
                  <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-purple-500 animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl">🧠</span>
                  </div>
                </div>
                <p className="text-gray-600 font-medium">Generating XAI Heatmap</p>
                <p className="text-xs text-gray-400 mt-2 max-w-xs mx-auto">
                  The explainable AI model is analyzing which parts of the leaf contributed to the diagnosis
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

    </div>
  )
}