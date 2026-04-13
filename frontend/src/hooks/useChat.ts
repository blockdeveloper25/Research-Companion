/**
 * useChat — manages the message list for one session and handles SSE streaming.
 *
 * How streaming works:
 *   1. User submits a question.
 *   2. We add a user message immediately (optimistic).
 *   3. We add an empty assistant message marked isStreaming=true.
 *   4. We POST to /api/chat and read the response body as a stream.
 *   5. Each "data: {...}" line arrives — token events append text,
 *      the done event attaches sources + confidence.
 *   6. isStreaming is set to false when done.
 *
 * Why fetch instead of EventSource?
 *   EventSource only supports GET. Our chat endpoint is POST (needs a body),
 *   so we use fetch + ReadableStream and parse SSE format manually.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Message, Source, Confidence } from '../types'
import * as api from '../lib/api'

export function useChat(sessionId: string) {
  const [messages, setMessages]     = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError]           = useState<string | null>(null)

  // Keep folders in a ref so the latest value is always captured in sendMessage
  // without making sendMessage stale if folders change between renders.
  const foldersRef = useRef<string[]>([])

  // Load persisted messages whenever the session changes
  useEffect(() => {
    setMessages([])
    setError(null)
    api.getMessages(sessionId).then(apiMessages => {
      setMessages(
        apiMessages.map(m => ({
          id:          m.id,
          role:        m.role,
          content:     m.content,
          sources:     m.sources,
          confidence:  m.confidence,
          model_used:  m.model_used,
        }))
      )
    }).catch(err => {
      setError(`Failed to load messages: ${err.message}`)
    })
  }, [sessionId])

  /** Called by App when the user changes the folder scope */
  const setFolders = useCallback((folders: string[]) => {
    foldersRef.current = folders
  }, [])

  /**
   * Send one question and stream the answer.
   * Returns the first line of the answer (used to auto-title the session).
   */
  const sendMessage = useCallback(async (
    question: string,
    onTitle?: (title: string) => void,
  ): Promise<void> => {
    if (isStreaming || !question.trim()) return

    const userMsgId      = crypto.randomUUID()
    const assistantMsgId = crypto.randomUUID()

    // Add user message immediately
    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user', content: question, sources: [], confidence: null, model_used: '' },
    ])

    // Add a blank assistant message that will fill in as tokens arrive
    setMessages(prev => [
      ...prev,
      { id: assistantMsgId, role: 'assistant', content: '', sources: [], confidence: null, model_used: '', isStreaming: true },
    ])

    setIsStreaming(true)
    setError(null)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          question,
          folders: foldersRef.current,
        }),
      })

      if (!res.ok) {
        throw new Error(`Server error ${res.status}`)
      }

      const reader  = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer    = ''
      let firstToken = true

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // Append incoming bytes to buffer, then process all complete lines.
        // We keep the last (possibly incomplete) line in the buffer.
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue

          let event: { type: string; content?: string; sources?: Source[]; confidence?: Confidence; model_used?: string; message?: string }
          try {
            event = JSON.parse(line.slice(6))
          } catch {
            continue
          }

          if (event.type === 'token' && event.content !== undefined) {
            // Auto-title: use the first token batch as the session title
            if (firstToken && onTitle) {
              onTitle(question)
              firstToken = false
            }
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantMsgId
                  ? { ...m, content: m.content + event.content }
                  : m
              )
            )
          } else if (event.type === 'done') {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      sources:     event.sources     ?? [],
                      confidence:  event.confidence  ?? null,
                      model_used:  event.model_used  ?? '',
                      isStreaming: false,
                    }
                  : m
              )
            )
          } else if (event.type === 'error') {
            setError(event.message ?? 'Unknown error from server')
            setMessages(prev => prev.filter(m => m.id !== assistantMsgId))
          }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unexpected error'
      setError(msg)
      // Remove the empty assistant bubble on failure
      setMessages(prev => prev.filter(m => m.id !== assistantMsgId))
    } finally {
      setIsStreaming(false)
    }
  }, [sessionId, isStreaming])

  const clearMessages = useCallback(async () => {
    await api.clearSession(sessionId)
    setMessages([])
  }, [sessionId])

  return { messages, isStreaming, error, sendMessage, setFolders, clearMessages }
}
