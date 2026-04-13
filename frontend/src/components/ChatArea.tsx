/**
 * ChatArea.tsx — The main conversation panel.
 *
 * Layout:
 *   ┌─────────────────────────────────────────┐
 *   │ Header (folder scope + clear button)    │
 *   ├─────────────────────────────────────────┤
 *   │ Message list (scrollable, flex-1)       │
 *   ├─────────────────────────────────────────┤
 *   │ Input area (fixed bottom)               │
 *   └─────────────────────────────────────────┘
 */

import { useEffect, useRef, useState } from 'react'
import { Send, Trash2, AlertCircle } from 'lucide-react'
import { useChat } from '../hooks/useChat'
import MessageBubble from './MessageBubble'

interface Props {
  sessionId:       string
  folders:         string[]
  onTitleUpdate:   (title: string) => void
  // onDeleteSession: () => void  // TODO: implement delete session UI
}

export default function ChatArea({ sessionId, folders, onTitleUpdate }: Props) {
  const { messages, isStreaming, error, sendMessage, setFolders, clearMessages } = useChat(sessionId)
  const [input, setInput] = useState('')
  const bottomRef   = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Keep the hook's folder ref in sync with the prop
  useEffect(() => {
    setFolders(folders)
  }, [folders, setFolders])

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Auto-resize textarea as the user types
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`
  }, [input])

  const handleSubmit = async () => {
    const q = input.trim()
    if (!q || isStreaming) return
    setInput('')
    await sendMessage(q, onTitleUpdate)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleClear = async () => {
    if (window.confirm('Clear all messages in this conversation?')) {
      await clearMessages()
    }
  }

  return (
    <div className="flex flex-col h-full bg-surface">

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-border shrink-0">
        <span className="text-sm text-muted select-none">
          {folders.length > 0
            ? `Searching: ${folders.join(', ')}`
            : 'All folders'}
        </span>
        <button
          onClick={handleClear}
          title="Clear conversation"
          className="p-1.5 rounded-lg text-muted hover:text-black hover:bg-raised transition-colors"
        >
          <Trash2 size={15} />
        </button>
      </header>

      {/* ── Message list ────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto flex flex-col gap-6">

          {messages.length === 0 && !isStreaming && (
            <div className="flex flex-col items-center justify-center h-64 gap-3 text-muted select-none">
              <p className="text-lg font-light">How can I help?</p>
              <p className="text-sm text-dim">Ask anything about your documents.</p>
            </div>
          )}

          {messages.map(message => (
            <MessageBubble key={message.id} message={message} />
          ))}

          {/* Error banner */}
          {error && (
            <div className="flex items-center gap-2 px-4 py-3 rounded-lg border border-red-200 bg-red-50 text-red-600 text-sm">
              <AlertCircle size={15} className="shrink-0" />
              {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── Input area ──────────────────────────────────────────────────────── */}
      <div className="shrink-0 px-4 pb-4 bg-surface">
        <div className="max-w-2xl mx-auto">
          <div className="flex items-end gap-3 bg-raised border border-border rounded-2xl px-4 py-3
                          focus-within:border-black/20 transition-colors">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Knowledge Companion…"
              rows={1}
              disabled={isStreaming}
              className="flex-1 bg-transparent text-sm text-black placeholder-dim resize-none
                         outline-none leading-relaxed min-h-[24px] max-h-36 disabled:opacity-50"
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isStreaming}
              className="shrink-0 p-1.5 rounded-lg bg-black text-white
                         disabled:opacity-30 disabled:cursor-not-allowed
                         hover:bg-black/80 transition-colors"
              title="Send (Enter)"
            >
              <Send size={15} />
            </button>
          </div>
          <p className="text-center text-xs text-dim mt-2 select-none">
            Shift+Enter for new line · all answers come from your documents only
          </p>
        </div>
      </div>
    </div>
  )
}
