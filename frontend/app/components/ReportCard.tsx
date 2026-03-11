'use client'

import ReactMarkdown from 'react-markdown'

interface ReportCardProps {
  diseaseName: string
  symptoms: string[]
  treatments: string[]
  report: string
  confidence?: number
}

export default function ReportCard({
  diseaseName,
  symptoms,
  treatments,
  report,
  confidence
}: ReportCardProps) {
  return (
    <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-2xl font-bold text-green-800">
          📋 Detailed Report for {diseaseName}
        </h3>
        {confidence && (
          <div className="text-right">
            <span className="text-sm text-gray-600">Confidence</span>
            <div className="text-lg font-bold text-green-700">
              {Math.round(confidence * 100)}%
            </div>
          </div>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div>
          <h4 className="font-semibold text-red-700 mb-2">⚠️ Symptoms</h4>
          <ul className="space-y-1">
            {symptoms.map((symptom, index) => (
              <li key={index} className="flex items-start">
                <span className="text-red-500 mr-2">•</span>
                <span className="text-gray-700">{symptom}</span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h4 className="font-semibold text-blue-700 mb-2">💊 Treatments</h4>
          <ul className="space-y-1">
            {treatments.map((treatment, index) => (
              <li key={index} className="flex items-start">
                <span className="text-blue-500 mr-2">✓</span>
                <span className="text-gray-700">{treatment}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-purple-700 mb-2">📝 AI Expert Analysis</h4>
        <div className="bg-white p-4 rounded-lg border min-h-[100px] flex flex-col justify-center">
          {report === 'Generating expert report and diagnosis...' ? (
            <div className="flex items-center space-x-3 text-gray-400 italic">
              <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
              <span>Consulting X-AI...</span>
            </div>
          ) : (
            <div className="prose prose-sm max-w-none text-gray-800">
              <ReactMarkdown>{report}</ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
