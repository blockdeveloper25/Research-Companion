/**
 * api.ts — Typed client for all non-streaming API calls.
 * Streaming (SSE) is handled separately in hooks/useChat.ts.
 */

import type { ApiMessage, Session, GraphEntity, GraphRelationship, GraphStats } from '../types'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export function listSessions(): Promise<Session[]> {
  return request<Session[]>('/sessions')
}

export function createSession(folders: string[] = [], title = 'New Chat'): Promise<Session> {
  return request<Session>('/sessions', {
    method: 'POST',
    body: JSON.stringify({ folders, title }),
  })
}

export function deleteSession(id: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/sessions/${id}`, { method: 'DELETE' })
}

export function clearSession(id: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/sessions/${id}/clear`, { method: 'POST' })
}

export function updateFolders(id: string, folders: string[]): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/sessions/${id}/folders`, {
    method: 'PATCH',
    body: JSON.stringify({ folders }),
  })
}

// ── Messages ──────────────────────────────────────────────────────────────────

export function getMessages(sessionId: string): Promise<ApiMessage[]> {
  return request<ApiMessage[]>(`/sessions/${sessionId}/messages`)
}

// ── Folders ───────────────────────────────────────────────────────────────────

export function listFolders(): Promise<string[]> {
  return request<string[]>('/folders')
}

export interface DocumentCategory {
  categories: string[]
  descriptions: Record<string, string>
}

export function getDocumentCategories(): Promise<DocumentCategory> {
  return request<DocumentCategory>('/document-categories')
}

// ── Documents ──────────────────────────────────────────────────────────────────

export interface Document {
  id: string
  filename: string
  folder: string
  chunks: number
  pages: number
  entities: number
  relationships: number
  ingested_at: string
}

export function listDocuments(): Promise<Document[]> {
  return request<Document[]>('/documents')
}

export function deleteDocument(filename: string): Promise<{ ok: boolean; message: string }> {
  return request<{ ok: boolean; message: string }>(`/documents/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  })
}

export function uploadDocuments(
  files: File[],
  folder: string = 'uploaded',
): Promise<{ ok: boolean; message: string; processed?: number; failed?: number; queued?: number; files: string[]; folder: string; status?: string }> {
  const formData = new FormData()
  formData.append('folder', folder)
  files.forEach(file => {
    formData.append('files', file)
  })
  
  return fetch(`${BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  }).then(async res => {
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new Error(`API ${res.status}: ${text}`)
    }
    return res.json()
  })
}

export function deleteFolder(folderName: string): Promise<{ ok: boolean; message: string; deleted_count: number }> {
  return request<{ ok: boolean; message: string; deleted_count: number }>(
    `/folders/${encodeURIComponent(folderName)}`,
    { method: 'DELETE' },
  )
}

export function getFolderDocuments(folderName: string): Promise<Document[]> {
  return request<Document[]>(`/folders/${encodeURIComponent(folderName)}/documents`)
}

export function syncDocuments(): Promise<{ ok: boolean; message: string; count: number }> {
  return request<{ ok: boolean; message: string; count: number }>('/documents/sync', {
    method: 'POST',
  })
}

export interface DebugInfo {
  database: {
    total_chunks: number
    unique_documents: number
    documents: Array<{
      doc_id: string
      filename: string
      folder: string
      chunks: number
    }>
  }
  ingestion_log: {
    file_exists?: boolean
    document_count?: number
    documents?: Array<{
      id: string
      filename: string
      folder: string
      chunks: number
    }>
    error?: string
  }
  error?: string
}

export function getDebugInfo(): Promise<DebugInfo> {
  return request<DebugInfo>('/debug/database-info')
}

// ── Knowledge Graph ──────────────────────────────────────────────────────────

export function listEntities(params: {
  search?: string
  type?: string
  folder?: string
  limit?: number
  offset?: number
} = {}): Promise<GraphEntity[]> {
  const query = new URLSearchParams()
  if (params.search) query.set('search', params.search)
  if (params.type)   query.set('type', params.type)
  if (params.folder) query.set('folder', params.folder)
  if (params.limit)  query.set('limit', String(params.limit))
  if (params.offset) query.set('offset', String(params.offset))
  const qs = query.toString()
  return request<GraphEntity[]>(`/graph/entities${qs ? `?${qs}` : ''}`)
}

export function getEntity(id: string): Promise<{
  entity: GraphEntity
  relationships: GraphRelationship[]
}> {
  return request(`/graph/entities/${id}`)
}

export function getRelatedEntities(
  id: string,
  depth = 2,
): Promise<GraphEntity[]> {
  return request<GraphEntity[]>(`/graph/entities/${id}/related?depth=${depth}`)
}

export function getGraphStats(): Promise<GraphStats> {
  return request<GraphStats>('/graph/stats')
}
