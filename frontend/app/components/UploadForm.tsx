'use client'

import { useRef } from 'react'

interface UploadFormProps {
  onUpload: (file: File) => void
  loading: boolean
}

export default function UploadForm({ onUpload, loading }: UploadFormProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0])
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div 
      className="border-2 border-dashed border-green-300 rounded-xl p-8 text-center hover:border-green-400 transition-colors cursor-pointer bg-green-50"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        className="hidden"
      />
      
      <div className="space-y-4">
        <div className="text-4xl">🍃</div>
        <h3 className="text-xl font-semibold text-green-800">
          Upload Mango Leaf Image
        </h3>
        <p className="text-gray-600">
          Drag & drop your image here, or click to browse
        </p>
        <p className="text-sm text-gray-500">
          Supported formats: JPG, PNG, WEBP (Max 5MB)
        </p>
        
        {loading && (
          <div className="flex items-center justify-center mt-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-600"></div>
            <span className="ml-2 text-green-700">Analyzing leaf...</span>
          </div>
        )}
      </div>
    </div>
  )
}
