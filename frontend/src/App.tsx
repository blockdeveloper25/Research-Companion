/**
 * App.tsx — Root layout: sidebar + chat area side by side.
 *
 * State that lives here (shared between sidebar and chat):
 *   - sessions list    (shown in sidebar, mutated by new chat / delete)
 *   - activeSessionId  (which conversation is open)
 *   - availableFolders (discovered from the documents/ directory)
 *   - selectedFolders  (which folders the current session searches)
 *   - sidebarOpen      (collapsed / expanded)
 *   - sidebarWidth     (drag-resizable, 180–480px)
 *   - documentBrowserOpen (whether the document manager modal is shown)
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import EntityBrowser from './components/EntityBrowser'
import DocumentBrowser from './components/DocumentBrowser'
import UploadModal from './components/UploadModal'
import type { Session } from './types'
import * as api from './lib/api'

const SIDEBAR_MIN = 180
const SIDEBAR_MAX = 480
const SIDEBAR_DEFAULT = 256

export default function App() {
  const [sessions,         setSessions]         = useState<Session[]>([])
  const [activeId,         setActiveId]         = useState<string | null>(null)
  const [availableFolders, setAvailableFolders] = useState<string[]>([])
  const [selectedFolders,  setSelectedFolders]  = useState<string[]>([])
  const [sidebarOpen,      setSidebarOpen]      = useState(true)
  const [sidebarWidth,     setSidebarWidth]     = useState(SIDEBAR_DEFAULT)
  const [activeView,       setActiveView]       = useState<'chat' | 'graph'>('chat')
  const [loading,          setLoading]          = useState(true)
  const [documentBrowserOpen, setDocumentBrowserOpen] = useState(false)
  const [uploadModalOpen, setUploadModalOpen] = useState(false)
  const [refreshDocuments, setRefreshDocuments] = useState(0)  // Trigger for sidebar document refresh

  // ── Bootstrap on first load ────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false

    async function loadWithRetry(retries = 15, delay = 1000) {
      for (let i = 0; i < retries; i++) {
        try {
          const [folders, sessions] = await Promise.all([api.listFolders(), api.listSessions()])
          if (cancelled) return
          setAvailableFolders(folders)
          setSessions(sessions)
          if (sessions.length > 0) {
            setActiveId(sessions[0].id)
            setSelectedFolders(sessions[0].folders)
          }
          setLoading(false)
          return
        } catch {
          if (cancelled) return
          await new Promise(r => setTimeout(r, delay))
        }
      }
      if (!cancelled) setLoading(false)
    }

    loadWithRetry()
    return () => { cancelled = true }
  }, [])

  // Sync selected folders whenever the active session changes
  useEffect(() => {
    const session = sessions.find(s => s.id === activeId)
    if (session) setSelectedFolders(session.folders)
  }, [activeId, sessions])

  // ── Sidebar resize drag ────────────────────────────────────────────────────

  const isDragging = useRef(false)

  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true
    e.preventDefault()
  }, [])

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return
      setSidebarWidth(Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, e.clientX)))
    }
    const onMouseUp = () => { isDragging.current = false }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [])

  // ── Session actions ────────────────────────────────────────────────────────

  const handleNewChat = useCallback(async () => {
    const session = await api.createSession(selectedFolders)
    setSessions(prev => [session, ...prev])
    setActiveId(session.id)
  }, [selectedFolders])

  const handleSelectSession = useCallback((id: string) => {
    setActiveId(id)
  }, [])

  const handleDeleteSession = useCallback(async (id: string) => {
    await api.deleteSession(id)
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id)
      if (activeId === id) {
        setActiveId(next.length > 0 ? next[0].id : null)
      }
      return next
    })
  }, [activeId])

  // ── Folder scope ──────────────────────────────────────────────────────────

  const handleFolderToggle = useCallback(async (folder: string, checked: boolean) => {
    const next = checked
      ? [...selectedFolders, folder]
      : selectedFolders.filter(f => f !== folder)

    setSelectedFolders(next)

    if (activeId) {
      await api.updateFolders(activeId, next)
      setSessions(prev =>
        prev.map(s => s.id === activeId ? { ...s, folders: next } : s)
      )
    }
  }, [selectedFolders, activeId])

  // ── Session title auto-update ─────────────────────────────────────────────

  const handleTitleUpdate = useCallback((id: string, title: string) => {
    setSessions(prev =>
      prev.map(s => s.id === id ? { ...s, title } : s)
    )
  }, [])

  // ── Render ────────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-surface">
        <div className="flex flex-col items-center gap-4 text-muted select-none">
          <div className="w-8 h-8 border-2 border-muted border-t-black rounded-full animate-spin" />
          <p className="text-sm tracking-wide">Connecting to Knowledge Companion...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-surface text-black overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        availableFolders={availableFolders}
        selectedFolders={selectedFolders}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onFolderToggle={handleFolderToggle}
        isOpen={sidebarOpen}
        width={sidebarWidth}
        activeView={activeView}
        onToggle={() => setSidebarOpen(prev => !prev)}
        onViewChange={setActiveView}
        onDocumentsClick={() => setDocumentBrowserOpen(true)}
        onUploadClick={() => setUploadModalOpen(true)}
        refreshTrigger={refreshDocuments}
      />

      {/* ── Resize handle — only when sidebar is open ─────────────────────── */}
      {sidebarOpen && (
        <div
          onMouseDown={handleResizeMouseDown}
          className="w-1 shrink-0 cursor-col-resize bg-border hover:bg-black/20 transition-colors"
          title="Drag to resize"
        />
      )}

      <main className="flex-1 overflow-hidden">
        {activeView === 'graph' ? (
          <EntityBrowser />
        ) : activeId ? (
          <ChatArea
            key={activeId}
            sessionId={activeId}
            folders={selectedFolders}
            onTitleUpdate={(title) => handleTitleUpdate(activeId, title)}
          />
        ) : (
          <EmptyState onNewChat={handleNewChat} />
        )}
      </main>

      {/* ── Document Browser Modal ─────────────────────────────────────────── */}
      <DocumentBrowser
        isOpen={documentBrowserOpen}
        onClose={() => setDocumentBrowserOpen(false)}
        onDocumentDeleted={() => {
          // Optionally refresh folders after deletion
          api.listFolders().then(setAvailableFolders).catch(console.error)
        }}
      />

      {/* ── Upload Modal ────────────────────────────────────────────────────── */}
      <UploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onUploadSuccess={() => {
          // Refresh documents and folders after successful upload
          api.listFolders().then(setAvailableFolders).catch(console.error)
          // Trigger sidebar to refresh documents list
          setRefreshDocuments(prev => prev + 1)
        }}
      />
    </div>
  )
}

function EmptyState({ onNewChat }: { onNewChat: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-muted select-none">
      <p className="text-xl font-light tracking-wide">Knowledge Companion</p>
      <p className="text-sm">Ask questions about your documents</p>
      <button
        onClick={onNewChat}
        className="mt-2 px-5 py-2 border border-border rounded-lg text-sm text-black
                   hover:bg-raised transition-colors"
      >
        Start a conversation
      </button>
    </div>
  )
}
