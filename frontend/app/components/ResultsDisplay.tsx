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
  const confidenceColor = confidence > 0.8 ? 'bg-green-500' :
    confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <div className="space-y-4">
        <h3 className="text-2xl font-bold text-gray-800">Diagnosis Results</h3>

        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="font-semibold text-gray-700">Prediction:</span>
            <span className="text-xl font-bold text-blue-700">{prediction}</span>
          </div>

          <div className="mb-2">
            <div className="flex text-black justify-between text-sm mb-1">
              <span>Confidence:</span>
              <span>{confidencePercentage}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${confidenceColor}`}
                style={{ width: `${confidencePercentage}%` }}
              ></div>
            </div>
          </div>

          {classProbabilities && (
            <div className="mt-4">
              <h4 className="font-semibold text-gray-700 mb-2">All Probabilities:</h4>
              <div className="space-y-1">
                {Object.entries(classProbabilities).map(([disease, prob]) => (
                  <div key={disease} className="flex justify-between text-sm">
                    <span className={disease === prediction ? 'font-bold text-blue-700' : 'font-semibold text-black'}>
                      {disease}
                    </span>
                    <span className={disease === prediction ? 'font-bold' : 'font-semibold'}>
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-2">AI Visual Explanation (Heatmap)</h3>
        <div className="border rounded-lg overflow-hidden bg-gray-100 min-h-[200px] flex items-center justify-center relative">
          {limeExplanation ? (
            <img
              src={`data:image/png;base64,${limeExplanation}`}
              alt="LIME explanation heatmap"
              className="w-full h-auto animate-in fade-in duration-500"
            />
          ) : (
            <div className="text-center p-6 bg-white w-full h-full flex flex-col items-center justify-center">
              <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
              <p className="text-sm text-gray-600 font-medium italic">
                Scanning leaf surface...<br />
                <span className="text-xs text-gray-400 font-normal">Building detailed XAI heatmap</span>
              </p>
            </div>
          )}
        </div>
        <p className="text-xs text-gray-500 mt-2">
          {limeExplanation
            ? "The highlighted areas show exactly where the AI detected disease patterns."
            : "LIME (Explainable AI) is currently analyzing which sections of the leaf caused this diagnosis."}
        </p>
      </div>
    </div>
  )
}
