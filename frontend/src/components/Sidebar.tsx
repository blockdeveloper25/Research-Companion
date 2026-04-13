/**
 * Sidebar.tsx — Left navigation panel.
 *
 * Three sections:
 *   1. Header + hamburger toggle + New Chat button
 *   2. Folder scope picker (checkboxes — which folders this session searches)
 *      - Expandable folders showing documents with delete options
 *   3. Conversation history list
 *
 * When collapsed (isOpen=false): renders a 48px strip with just the hamburger.
 * When open: renders at the width passed from App (drag-resizable).
 */

import { useState, useEffect } from 'react'
import { Trash2, Plus, FolderOpen, MessageSquare, Menu, Network, FileText, ChevronDown, ChevronRight, Upload } from 'lucide-react'
import { clsx } from 'clsx'
import type { Session } from '../types'
import type { Document } from '../lib/api'
import * as api from '../lib/api'

interface Props {
  sessions:         Session[]
  activeId:         string | null
  availableFolders: string[]
  selectedFolders:  string[]
  isOpen:           boolean
  width:            number
  activeView:       'chat' | 'graph'
  onNewChat:        () => void
  onSelectSession:  (id: string) => void
  onDeleteSession:  (id: string) => void
  onFolderToggle:   (folder: string, checked: boolean) => void
  onToggle:         () => void
  onViewChange:     (view: 'chat' | 'graph') => void
  onDocumentsClick: () => void
  onUploadClick:    () => void
  refreshTrigger:   number
}

export default function Sidebar({
  sessions,
  activeId,
  availableFolders,
  selectedFolders,
  isOpen,
  width,
  activeView,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onFolderToggle,
  onToggle,
  onViewChange,
  onDocumentsClick,
  onUploadClick,
  refreshTrigger,
}: Props) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
  const [documents, setDocuments] = useState<Document[]>([])
  const [deletingFile, setDeletingFile] = useState<string | null>(null)

  // ── Load documents on mount and when refresh is triggered ──────────────────
  useEffect(() => {
    if (!isOpen) return

    let cancelled = false

    async function loadDocuments() {
      try {
        // First sync the database with the ingestion log
        try {
          await api.syncDocuments()
        } catch (syncErr) {
          console.warn('Sync warning:', syncErr)
        }
        
        const docs = await api.listDocuments()
        if (!cancelled) {
          setDocuments(docs)
        }
      } catch (err) {
        console.error('Failed to load documents:', err)
      }
    }

    loadDocuments()
    return () => { cancelled = true }
  }, [isOpen, refreshTrigger])

  // ── Delete document handler ────────────────────────────────────────────────
  const handleDeleteDocument = async (filename: string) => {
    try {
      setDeletingFile(filename)
      await api.deleteDocument(filename)
      setDocuments(prev => prev.filter(d => d.filename !== filename))
    } catch (err) {
      console.error('Failed to delete document:', err)
    } finally {
      setDeletingFile(null)
    }
  }

  // ── Toggle folder expansion ────────────────────────────────────────────────
  const toggleFolderExpand = (folder: string) => {
    const next = new Set(expandedFolders)
    if (next.has(folder)) {
      next.delete(folder)
    } else {
      next.add(folder)
    }
    setExpandedFolders(next)
  }

  // ── Get documents for a specific folder ─────────────────────────────────────
  const getDocumentsInFolder = (folder: string) => {
    return documents.filter(d => d.folder === folder)
  }

  // ── Collapsed strip ───────────────────────────────────────────────────────
  if (!isOpen) {
    return (
      <aside className="flex flex-col items-center h-full bg-sidebar border-r border-border shrink-0" style={{ width: 48 }}>
        <button
          onClick={onToggle}
          className="mt-4 p-2 rounded-lg text-muted hover:text-black hover:bg-raised transition-colors"
          title="Expand sidebar"
        >
          <Menu size={18} />
        </button>
      </aside>
    )
  }

  // ── Expanded sidebar ──────────────────────────────────────────────────────
  return (
    <aside
      className="flex flex-col h-full bg-sidebar border-r border-border shrink-0 overflow-hidden"
      style={{ width }}
    >

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-4 pt-5 pb-3 border-b border-border">
        <p className="text-sm font-semibold tracking-widest uppercase text-muted select-none truncate">
          Knowledge Companion
        </p>
        <button
          onClick={onToggle}
          className="ml-2 p-1.5 rounded-lg text-muted hover:text-black hover:bg-raised transition-colors shrink-0"
          title="Collapse sidebar"
        >
          <Menu size={16} />
        </button>
      </div>

      {/* ── New Chat button ─────────────────────────────────────────────────── */}
      <div className="px-3 pt-3 flex gap-2">
        <button
          onClick={onNewChat}
          className="flex items-center gap-2 flex-1 px-3 py-2 rounded-lg border border-border
                     text-sm text-black hover:bg-raised transition-colors"
        >
          <Plus size={15} />
          New chat
        </button>
        <button
          onClick={onUploadClick}
          className="p-2 rounded-lg border border-border text-black hover:bg-raised transition-colors shrink-0"
          title="Upload documents"
        >
          <Upload size={15} />
        </button>
        <button
          onClick={onDocumentsClick}
          className="p-2 rounded-lg border border-border text-black hover:bg-raised transition-colors shrink-0"
          title="Manage documents"
        >
          <FileText size={15} />
        </button>
      </div>

      {/* ── View switcher (Chat / Graph) ──────────────────────────────────── */}
      <div className="flex gap-1 px-3 pt-3">
        <button
          onClick={() => onViewChange('chat')}
          className={clsx(
            'flex items-center gap-1.5 flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
            activeView === 'chat'
              ? 'bg-raised text-black'
              : 'text-muted hover:text-black hover:bg-raised/50',
          )}
        >
          <MessageSquare size={12} />
          Chat
        </button>
        <button
          onClick={() => onViewChange('graph')}
          className={clsx(
            'flex items-center gap-1.5 flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
            activeView === 'graph'
              ? 'bg-raised text-black'
              : 'text-muted hover:text-black hover:bg-raised/50',
          )}
        >
          <Network size={12} />
          Graph
        </button>
      </div>

      {/* ── Folder picker ───────────────────────────────────────────────────── */}
      {availableFolders.length > 0 && (
        <div className="px-4 pt-4 pb-2">
          <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-widest text-muted mb-2 select-none">
            <FolderOpen size={12} />
            Folders
          </p>
          <div className="flex flex-col gap-1">
            {availableFolders.map(folder => {
              const isExpanded = expandedFolders.has(folder)
              const docsInFolder = getDocumentsInFolder(folder)
              
              return (
                <div key={folder} className="flex flex-col gap-1">
                  {/* Folder checkbox and expand button */}
                  <div className="flex items-center gap-1">
                    {docsInFolder.length > 0 && (
                      <button
                        onClick={() => toggleFolderExpand(folder)}
                        className="p-0.5 rounded hover:bg-raised text-muted hover:text-black transition-colors shrink-0"
                        title={isExpanded ? 'Collapse' : 'Expand'}
                      >
                        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      </button>
                    )}
                    {docsInFolder.length === 0 && <div className="w-5" />}
                    
                    <label className="flex items-center gap-2 text-sm text-black cursor-pointer
                                     py-0.5 hover:text-black/70 transition-colors select-none flex-1">
                      <input
                        type="checkbox"
                        checked={selectedFolders.includes(folder)}
                        onChange={e => onFolderToggle(folder, e.target.checked)}
                        className="accent-black w-3.5 h-3.5 rounded cursor-pointer"
                      />
                      <span className="truncate">{folder}</span>
                      {docsInFolder.length > 0 && (
                        <span className="text-xs text-muted ml-auto shrink-0">
                          ({docsInFolder.length})
                        </span>
                      )}
                    </label>
                  </div>

                  {/* Expanded documents list */}
                  {isExpanded && docsInFolder.length > 0 && (
                    <div className="ml-6 border-l border-border pl-2 flex flex-col gap-1">
                      {docsInFolder.map(doc => (
                        <div
                          key={doc.id}
                          className="group flex items-center justify-between gap-2 text-xs py-1 px-2
                                     rounded hover:bg-raised transition-colors"
                        >
                          <div className="flex-1 min-w-0">
                            <p className="truncate text-black hover:text-black/70 cursor-default">
                              {doc.filename}
                            </p>
                            <p className="text-muted text-xs mt-0.5">
                              {doc.chunks} chunks
                            </p>
                          </div>
                          
                          {/* Delete button */}
                          <button
                            onClick={() => handleDeleteDocument(doc.filename)}
                            disabled={deletingFile === doc.filename}
                            className="p-1 rounded text-red-600 bg-red-50/80
                                     hover:bg-red-100 disabled:opacity-50 transition-colors shrink-0"
                            title="Delete document"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── Conversation history ─────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-2 py-3">
        {sessions.length === 0 ? (
          <p className="text-xs text-dim text-center mt-4 select-none">No conversations yet</p>
        ) : (
          <div className="flex flex-col gap-0.5">
            <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-widest text-muted mb-2 px-2 select-none">
              <MessageSquare size={12} />
              History
            </p>
            {sessions.map(session => (
              <SessionRow
                key={session.id}
                session={session}
                isActive={session.id === activeId}
                onSelect={() => onSelectSession(session.id)}
                onDelete={() => onDeleteSession(session.id)}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}

// ── Session row ───────────────────────────────────────────────────────────────

function SessionRow({
  session,
  isActive,
  onSelect,
  onDelete,
}: {
  session:  Session
  isActive: boolean
  onSelect: () => void
  onDelete: () => void
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={e => e.key === 'Enter' && onSelect()}
      className={clsx(
        'group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer',
        'transition-colors text-sm',
        isActive
          ? 'bg-raised text-black font-medium'
          : 'text-muted hover:bg-raised hover:text-black',
      )}
    >
      <span className="truncate flex-1">{session.title || 'New Chat'}</span>

      {/* Delete button — only visible on hover */}
      <button
        onClick={e => { e.stopPropagation(); onDelete() }}
        className="opacity-0 group-hover:opacity-100 ml-1 p-1 rounded
                   hover:text-black text-muted transition-all shrink-0"
        title="Delete conversation"
      >
        <Trash2 size={13} />
      </button>
    </div>
  )
}
