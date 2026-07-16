'use client'

import { useRef, useState } from 'react'

interface UploadFormProps {
  onUpload: (file: File) => void
  loading: boolean
}

export default function UploadForm({ onUpload, loading }: UploadFormProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      setSelectedFile(file)
      onUpload(file)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setSelectedFile(file)
      onUpload(file)
    }
  }

  const handleClick = () => {
    if (!loading) fileInputRef.current?.click()
  }

  return (
    <div className="text-center">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`
          relative cursor-pointer transition-all duration-300 rounded-2xl
          ${dragOver
            ? 'border-green-500 bg-green-50 scale-105'
            : 'border-green-300 hover:border-green-400 bg-gradient-to-b from-green-50/50 to-white'
          }
          ${loading ? 'pointer-events-none' : ''}
          border-2 border-dashed p-12 md:p-16
          group
        `}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          className="hidden"
        />

        {loading ? (
          <div className="space-y-6">
            {/* Loading Animation */}
            <div className="relative w-24 h-24 mx-auto">
              <div className="absolute inset-0 rounded-full border-4 border-green-100"></div>
              <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-green-500 animate-spin"></div>
              <div className="absolute inset-0 rounded-full border-4 border-transparent border-b-emerald-500 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '0.8s' }}></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-3xl animate-pulse">🔍</span>
              </div>
            </div>

            <div className="space-y-2">
              <h3 className="text-xl font-semibold text-green-800">
                Analyzing Leaf...
              </h3>
              <div className="max-w-md mx-auto space-y-2">
                <div className="h-2 bg-green-100 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full animate-shimmer" style={{ width: '60%' }}></div>
                </div>
                <div className="flex justify-center gap-1">
                  <span className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></span>
                  <span className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                  <span className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                </div>
              </div>
              <p className="text-sm text-gray-500 italic">
                Running Vision Transformer model...
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Upload Icon */}
            <div className="relative w-28 h-28 mx-auto">
              <div className="absolute inset-0 bg-gradient-to-br from-green-100 to-emerald-100 rounded-3xl rotate-6 group-hover:rotate-12 transition-transform"></div>
              <div className="absolute inset-0 bg-gradient-to-br from-green-200 to-emerald-200 rounded-3xl -rotate-3 group-hover:-rotate-6 transition-transform"></div>
              <div className="absolute inset-0 bg-white rounded-3xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow">
                <span className="text-5xl group-hover:scale-110 transition-transform">🍃</span>
              </div>
            </div>

            <div className="space-y-3">
              <h3 className="text-2xl font-bold text-gray-800 group-hover:text-green-700 transition-colors">
                Upload Mango Leaf Image
              </h3>
              <p className="text-gray-500 text-lg">
                Drag & drop your image here, or click to browse
              </p>
              <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
                <span className="px-3 py-1 bg-green-50 rounded-full text-green-600 font-medium">JPG</span>
                <span className="px-3 py-1 bg-blue-50 rounded-full text-blue-600 font-medium">PNG</span>
                <span className="px-3 py-1 bg-purple-50 rounded-full text-purple-600 font-medium">WEBP</span>
                <span className="text-gray-300">|</span>
                <span className="text-gray-400">Max 5MB</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {selectedFile && !loading && (
        <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-green-50 rounded-full text-sm text-green-700 animate-fade-in-up">
          <span>📎</span>
          <span className="font-medium truncate max-w-[200px]">{selectedFile.name}</span>
          <span className="text-green-400">•</span>
          <span>{(selectedFile.size / 1024 / 1024).toFixed(1)} MB</span>
        </div>
      )}
    </div>
  )
}