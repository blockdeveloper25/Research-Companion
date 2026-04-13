/**
 * DocumentBrowser.tsx — Modal/panel for viewing and managing ingested documents.
 *
 * Features:
 *   - List all ingested PDFs with metadata (folder, chunks, entities, etc)
 *   - Delete individual documents with confirmation
 *   - Delete entire folders with confirmation
 *   - Sort by ingestion date, filename, or folder
 *   - Filter by folder
 */

import { useEffect, useState } from 'react'
import { Trash2, FileText, X, FolderOpen } from 'lucide-react'
import { clsx } from 'clsx'
import type { Document } from '../lib/api'
import * as api from '../lib/api'

interface Props {
  isOpen: boolean
  onClose: () => void
  onDocumentDeleted?: () => void
}

type SortKey = 'name' | 'folder' | 'date' | 'chunks'

export default function DocumentBrowser({ isOpen, onClose, onDocumentDeleted }: Props) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<SortKey>('date')
  const [filterFolder, setFilterFolder] = useState<string | null>(null)
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const [deletingFolder, setDeletingFolder] = useState<string | null>(null)
  const [deleteFolderConfirm, setDeleteFolderConfirm] = useState<string | null>(null)
  const [folderDocCount, setFolderDocCount] = useState<Record<string, number>>({})

  // ── Load documents on mount or when modal opens ─────────────────────────────

  useEffect(() => {
    if (!isOpen) return

    let cancelled = false

    async function loadDocuments() {
      try {
        setLoading(true)
        setError(null)
        
        // First, sync the database with the ingestion log
        // This ensures any documents ingested via CLI are visible in the UI
        try {
          await api.syncDocuments()
        } catch (syncErr) {
          // Sync failure is not critical, continue trying to load
          console.warn('Sync warning:', syncErr)
        }
        
        const docs = await api.listDocuments()
        if (!cancelled) {
          setDocuments(docs)
          
          // Calculate document count per folder
          const counts: Record<string, number> = {}
          docs.forEach(doc => {
            counts[doc.folder] = (counts[doc.folder] || 0) + 1
          })
          setFolderDocCount(counts)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load documents')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    loadDocuments()
    return () => { cancelled = true }
  }, [isOpen])

  // ── Sort and filter logic ──────────────────────────────────────────────────

  const folders = Array.from(new Set(documents.map(d => d.folder)))

  let filtered = documents
  if (filterFolder) {
    filtered = filtered.filter(d => d.folder === filterFolder)
  }

  const sorted = [...filtered]
  if (sortBy === 'name') {
    sorted.sort((a, b) => a.filename.localeCompare(b.filename))
  } else if (sortBy === 'folder') {
    sorted.sort((a, b) => a.folder.localeCompare(b.folder))
  } else if (sortBy === 'date') {
    sorted.sort((a, b) => new Date(b.ingested_at).getTime() - new Date(a.ingested_at).getTime())
  } else if (sortBy === 'chunks') {
    sorted.sort((a, b) => b.chunks - a.chunks)
  }

  // ── Delete document handler ─────────────────────────────────────────────────

  const handleDeleteDocument = async (filename: string) => {
    try {
      setDeletingFilename(filename)
      await api.deleteDocument(filename)
      setDocuments(prev => prev.filter(d => d.filename !== filename))
      setDeleteConfirm(null)
      setSuccess(`Deleted "${filename}"`)
      setTimeout(() => setSuccess(null), 3000)
      onDocumentDeleted?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document')
    } finally {
      setDeletingFilename(null)
    }
  }

  // ── Delete folder handler ──────────────────────────────────────────────────

  const handleDeleteFolder = async (folderName: string) => {
    try {
      setDeletingFolder(folderName)
      const result = await api.deleteFolder(folderName)
      setDocuments(prev => prev.filter(d => d.folder !== folderName))
      setDeleteFolderConfirm(null)
      setFilterFolder(null)
      setSuccess(`Deleted folder "${folderName}" with ${result.deleted_count} document(s)`)
      setTimeout(() => setSuccess(null), 3000)
      onDocumentDeleted?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete folder')
    } finally {
      setDeletingFolder(null)
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
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden
                      flex flex-col">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <FileText size={20} />
            <h2 className="text-lg font-semibold">Documents ({documents.length})</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-raised text-muted hover:text-black transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Error message */}
        {error && (
          <div className="px-6 py-3 bg-red-50 border-b border-red-200 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Success message */}
        {success && (
          <div className="px-6 py-3 bg-green-50 border-b border-green-200 text-sm text-green-700">
            {success}
          </div>
        )}

        {/* Folder management section */}
        {!loading && folders.length > 0 && (
          <div className="px-6 py-3 border-b border-border bg-raised">
            <p className="text-xs font-medium uppercase tracking-widest text-muted mb-2">Folders</p>
            <div className="flex flex-wrap gap-2">
              {folders.map(folder => (
                <div
                  key={folder}
                  className={clsx(
                    'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
                    filterFolder === folder
                      ? 'bg-black text-white'
                      : 'bg-white border border-border text-black hover:bg-white/80'
                  )}
                >
                  <button
                    onClick={() => setFilterFolder(filterFolder === folder ? null : folder)}
                    className="flex items-center gap-1 flex-1"
                    title={`Show documents in ${folder}`}
                  >
                    <FolderOpen size={14} />
                    <span className="truncate">{folder}</span>
                    <span className={clsx(
                      'text-xs ml-1 px-1.5 py-0.5 rounded',
                      filterFolder === folder ? 'bg-white/20' : 'bg-gray-100 text-gray-700'
                    )}>
                      {folderDocCount[folder] || 0}
                    </span>
                  </button>
                  
                  {/* Delete folder button */}
                  {deleteFolderConfirm === folder ? (
                    <div className="flex gap-1">
                      <button
                        onClick={() => handleDeleteFolder(folder)}
                        disabled={deletingFolder === folder}
                        className="px-2 py-0.5 text-xs bg-red-600 text-white rounded
                                   hover:bg-red-700 disabled:opacity-50 transition-colors"
                      >
                        {deletingFolder === folder ? '...' : 'Yes'}
                      </button>
                      <button
                        onClick={() => setDeleteFolderConfirm(null)}
                        className="px-2 py-0.5 text-xs bg-white/20 text-white rounded
                                   hover:bg-white/30 transition-colors"
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setDeleteFolderConfirm(folder)}
                      className="p-1 rounded hover:bg-red-50 text-muted hover:text-red-600
                                 transition-colors shrink-0"
                      title={`Delete "${folder}" folder (${folderDocCount[folder] || 0} files)`}
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Controls */}
        {!loading && documents.length > 0 && (
          <div className="px-6 py-3 border-b border-border bg-raised space-y-3">
            <div className="flex gap-4 flex-wrap">
              {/* Sort dropdown */}
              <div className="flex items-center gap-2">
                <label className="text-sm text-muted">Sort by:</label>
                <select
                  value={sortBy}
                  onChange={e => setSortBy(e.target.value as SortKey)}
                  className="px-2 py-1 text-sm border border-border rounded
                             bg-white text-black cursor-pointer"
                >
                  <option value="date">Ingestion date</option>
                  <option value="name">Filename</option>
                  <option value="folder">Folder</option>
                  <option value="chunks">Chunk count</option>
                </select>
              </div>
            </div>
            {filterFolder && (
              <p className="text-xs text-muted">
                Filtering: <span className="font-medium text-black">{filterFolder}</span>
              </p>
            )}
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-32 text-muted">
              Loading documents...
            </div>
          ) : sorted.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-muted">
              No documents found
            </div>
          ) : (
            <div className="divide-y divide-border">
              {sorted.map(doc => (
                <div
                  key={doc.id}
                  className="px-6 py-4 hover:bg-raised transition-colors group"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-black truncate">
                        {doc.filename}
                      </h3>
                      <div className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
                        <span>📁 {doc.folder}</span>
                        <span>📄 {doc.pages} pages</span>
                        <span>🔗 {doc.chunks} chunks</span>
                        <span>🏷️ {doc.entities} entities</span>
                        <span className="text-gray-400">
                          {new Date(doc.ingested_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>

                    {/* Delete button */}
                    {deleteConfirm === doc.filename ? (
                      <div className="flex gap-2 shrink-0">
                        <button
                          onClick={() =>
                            handleDeleteDocument(doc.filename)
                          }
                          disabled={deletingFilename === doc.filename}
                          className="px-3 py-1 text-sm bg-red-600 text-white rounded
                                     hover:bg-red-700 disabled:opacity-50 transition-colors"
                        >
                          {deletingFilename === doc.filename ? 'Deleting...' : 'Confirm'}
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(null)}
                          className="px-3 py-1 text-sm border border-border rounded
                                     hover:bg-raised transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setDeleteConfirm(doc.filename)}
                        className="p-2 rounded text-red-600 bg-red-50
                                   hover:bg-red-100 transition-colors shrink-0"
                        title="Delete document"
                      >
                        <Trash2 size={18} />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-border bg-raised flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium rounded border border-border
                       text-black hover:bg-raised transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
