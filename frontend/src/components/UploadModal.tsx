/**
 * UploadModal.tsx — Modal for uploading PDF files to the knowledge base.
 *
 * Features:
 *   - Drag-and-drop file upload
 *   - Select files via file picker
 *   - Shows which files are selected
 *   - Upload progress and status
 *   - Allows specifying folder name
 */

import { useRef, useState } from 'react'
import { Upload, X, AlertCircle, CheckCircle } from 'lucide-react'
import * as api from '../lib/api'

interface Props {
  isOpen: boolean
  onClose: () => void
  onUploadSuccess?: () => void
}

export default function UploadModal({ isOpen, onClose, onUploadSuccess }: Props) {
  const [files, setFiles] = useState<File[]>([])
  const [folderName, setFolderName] = useState('uploaded')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dragOverRef = useRef(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragOverRef.current = true
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragOverRef.current = false
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragOverRef.current = false

    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      file => file.type === 'application/pdf' || file.name.endsWith('.pdf'),
    )

    if (droppedFiles.length > 0) {
      setFiles(prev => [...prev, ...droppedFiles])
      setError(null)
    } else {
      setError('Only PDF files are supported')
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || [])
    if (selectedFiles.length > 0) {
      setFiles(prev => [...prev, ...selectedFiles])
      setError(null)
    }
  }

  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('No files selected')
      return
    }

    try {
      setUploading(true)
      setError(null)
      setSuccess(false)
      setUploadStatus('Uploading files...')

      const result = await api.uploadDocuments(files, folderName)

      if (result.ok) {
        setSuccess(true)
        // Show the processing message from the backend
        setUploadStatus(
          result.status === 'processing'
            ? `Queued ${result.queued} file(s) for background ingestion`
            : `Ingested ${result.processed || result.queued} file(s)`
        )
        setFiles([])
        setFolderName('uploaded')
        onUploadSuccess?.()
        
        // Auto close after 3 seconds but keep uploading flag true for background processing message
        setTimeout(() => {
          onClose()
          setSuccess(false)
          setUploadStatus('')
          setUploading(false)
        }, 3000)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      setUploading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center"
      onClick={e => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full max-h-[90vh] overflow-hidden
                      flex flex-col">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Upload size={20} />
            <h2 className="text-lg font-semibold">Upload PDFs</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-raised text-muted hover:text-black transition-colors"
            title={uploading ? 'Close modal (processing will continue)' : 'Close'}
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {/* Folder name input */}
          <div>
            <label className="block text-sm font-medium text-black mb-2">
              Folder Name
            </label>
            <input
              type="text"
              value={folderName}
              onChange={e => setFolderName(e.target.value)}
              disabled={uploading}
              placeholder="e.g., Research Papers"
              className="w-full px-3 py-2 border border-border rounded-lg bg-white text-black
                         placeholder-muted focus:outline-none focus:ring-2 focus:ring-black
                         disabled:opacity-50"
            />
            <p className="text-xs text-muted mt-1">
              Documents will be organized under this folder name
            </p>
          </div>

          {/* Drop zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors
                       ${dragOverRef.current ? 'border-black bg-black/5' : 'border-border hover:border-black/50'}
                       ${uploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <Upload size={32} className="mx-auto text-muted mb-2" />
            <p className="font-medium text-black mb-1">Drag and drop PDFs here</p>
            <p className="text-sm text-muted mb-3">or</p>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="px-4 py-2 bg-black text-white rounded-lg hover:bg-black/90
                         transition-colors text-sm font-medium disabled:opacity-50"
            >
              Select Files
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf"
              onChange={handleFileSelect}
              disabled={uploading}
              className="hidden"
            />
          </div>

          {/* Error message */}
          {error && (
            <div className="flex gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
              <AlertCircle size={18} className="text-red-600 shrink-0 mt-0.5" />
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {/* Success message */}
          {success && (
            <div className="flex gap-2 p-3 bg-green-50 border border-green-200 rounded-lg">
              <CheckCircle size={18} className="text-green-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm text-green-700 font-medium">{uploadStatus}</p>
                {uploading && (
                  <p className="text-xs text-green-600 mt-1">
                    Files are being processed in the background. You can close this modal and check the file manager later.
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Selected files list */}
          {files.length > 0 && !success && (
            <div>
              <p className="text-sm font-medium text-black mb-2">
                Selected Files ({files.length})
              </p>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {files.map((file, index) => (
                  <div
                    key={`${file.name}-${index}`}
                    className="flex items-center justify-between p-2 bg-raised rounded-lg border border-border"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-black truncate">{file.name}</p>
                      <p className="text-xs text-muted">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                    <button
                      onClick={() => handleRemoveFile(index)}
                      disabled={uploading}
                      className="ml-2 p-1 rounded hover:bg-white text-muted hover:text-black
                                transition-colors disabled:opacity-50 shrink-0"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Upload status */}
          {uploading && (
            <div className="flex items-center gap-2 text-sm text-muted">
              <div className="w-4 h-4 border-2 border-muted border-t-black rounded-full animate-spin" />
              {uploadStatus}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-border bg-raised flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium rounded border border-border
                       text-black hover:bg-white transition-colors"
          >
            {uploading ? 'Continue Working' : 'Cancel'}
          </button>
          <button
            onClick={handleUpload}
            disabled={uploading || files.length === 0}
            className="px-4 py-2 text-sm font-medium rounded bg-black text-white
                       hover:bg-black/90 transition-colors disabled:opacity-50"
          >
            {uploading ? 'Uploading...' : 'Upload & Ingest'}
          </button>
        </div>
      </div>
    </div>
  )
}
