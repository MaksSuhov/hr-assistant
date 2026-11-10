import { useCallback, useEffect, useMemo, useState } from "react";
import type { ChangeEvent, DragEvent, FormEvent, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

type AuthUser = { id: string; email: string; full_name: string; role: string };
let csrfToken = "";
function readCsrfToken() {
  return document.cookie.split("; ").find((part) => part.startsWith("csrf_token="))?.split("=").slice(1).join("=") ?? "";
}
async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}, retry = true): Promise<Response> {
  csrfToken = readCsrfToken() || csrfToken;
  const headers = new Headers(init.headers);
  if (csrfToken && !["GET", "HEAD", "OPTIONS"].includes((init.method ?? "GET").toUpperCase())) headers.set("X-CSRF-Token", csrfToken);
  const response = await fetch(input, { ...init, headers, credentials: "include" });
  if (response.status === 401 && retry && !String(input).includes("/auth/")) {
    const refreshed = await fetch(API + "/auth/refresh", { method: "POST", credentials: "include", headers: { "X-CSRF-Token": readCsrfToken() || csrfToken } });
    if (refreshed.ok) { csrfToken = readCsrfToken() || csrfToken; return apiFetch(input, init, false); }
  }
  return response;
}

type CandidateDocument = {
  id: string;
  filename: string;
  source: string;
  uploaded_by: string;
  uploaded_at: string;
  indexing_status: string;
  indexing_error: string | null;
  extraction_method: string;
};

type Application = {
  id: string;
  candidate_id: string;
  candidate_name: string | null;
  vacancy_id: string;
  vacancy_title: string;
  stage: string;
  stage_changed_at: string;
  created_at: string;
  updated_at: string;
};

type Candidate = {
  id: string;
  full_name: string | null;
  email: string | null;
  phone: string | null;
  created_at: string;
  updated_at: string;
  sources: CandidateDocument[];
  applications: Application[];
};

type Vacancy = {
  id: string;
  title: string;
  department: string | null;
  description: string;
  status: string;
  requirements: { id: string; text: string; is_must_have: boolean; weight: number }[];
};

type ChatMessage = { id: string; role: "user" | "assistant"; content: string; streaming?: boolean };
type View = "overview" | "candidates" | "vacancies" | "inbox";

const statusLabels: Record<string, string> = {
  new: "Новый",
  screening: "Скрининг",
  interview: "Интервью",
  offer: "Оффер",
  hired: "Нанят",
  rejected: "Архив",
  open: "Открыта",
  paused: "На паузе",
  closed: "Закрыта",
};

function candidateStage(candidate: Candidate) {
  return candidate.applications[0]?.stage ?? "new";
}

function Glyph({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, ReactNode> = {
    home: <><path d="m3 10 9-7 9 7"/><path d="M5 9v11h14V9M9 20v-6h6v6"/></>,
    people: <><path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="10" cy="7" r="4"/><path d="M20 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></>,
    briefcase: <><rect x="3" y="7" width="18" height="14" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M3 12h18M10 12v2h4v-2"/></>,
    inbox: <><path d="M4 4h16l1 11-4 5H7l-4-5 1-11Z"/><path d="M3 14h5l2 3h4l2-3h5"/></>,
    sparkle: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z"/><path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z"/></>,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6"/></>,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5"/><path d="M4 16v4h16v-4"/></>,
    link: <><path d="M10 13a5 5 0 0 0 7.1 0l3-3A5 5 0 0 0 13 2.9l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.1 0l-3 3A5 5 0 0 0 11 21.1l1.7-1.7"/></>,
    close: <><path d="m18 6-12 12M6 6l12 12"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4 20-7Z"/><path d="M22 2 11 13"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    check: <path d="m5 12 4 4L19 6"/>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h8"/></>,
    alert: <><path d="M10.3 3.9 2.4 17.6A2 2 0 0 0 4.1 20h15.8a2 2 0 0 0 1.7-2.4L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 4h.01"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] ?? paths.sparkle}</svg>;
}

function initials(name: string | null) {
  if (!name) return "?";
  return name.split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

function dateLabel(value: string) {
  return new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "short" }).format(new Date(value));
}

function normalizeAssistantMarkdown(content: string) {
  return content
    .replace(/\r\n?/g, "\n")
    .replace(/^( {0,6})(#{1,6})[ \t]*(?=\S)/gm, "$1$2 ");
}

export default function App() {
  const isCandidatePortal = window.location.pathname === "/candidate";
  const [view, setView] = useState<View>("overview");
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [vacancies, setVacancies] = useState<Vacancy[]>([]);
  const [apiError, setApiError] = useState(false);
  const [chatConfigured, setChatConfigured] = useState<boolean | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatDraft, setChatDraft] = useState("");
  const [chatPending, setChatPending] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  const [showVacancyForm, setShowVacancyForm] = useState(false);
  const [creatingVacancy, setCreatingVacancy] = useState(false);
  const [toast, setToast] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadName, setUploadName] = useState("");
  const [uploadSource, setUploadSource] = useState("Почта отдела кадров");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [linkValue, setLinkValue] = useState("");
  const [linkContext, setLinkContext] = useState("");
  const [selectedVacancyId, setSelectedVacancyId] = useState("");
  const [portalName, setPortalName] = useState("");
  const [portalFile, setPortalFile] = useState<File | null>(null);
  const [portalSubmitted, setPortalSubmitted] = useState(false);
  const [activeTab, setActiveTab] = useState<"file" | "link">("file");
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authName, setAuthName] = useState("");
  const [authMessage, setAuthMessage] = useState("");
  const [authBusy, setAuthBusy] = useState(false);

  const refreshData = useCallback(async () => {
    try {
      const [candidateResponse, vacancyResponse, healthResponse] = await Promise.all([
        apiFetch(`${API}/candidates`),
        apiFetch(`${API}/vacancies`),
        apiFetch(`${API}/health`),
      ]);
      if (!candidateResponse.ok || !vacancyResponse.ok) throw new Error("API unavailable");
      setCandidates(await candidateResponse.json());
      setVacancies(await vacancyResponse.json());
      if (healthResponse.ok) {
        const health = await healthResponse.json();
        setChatConfigured(health.openai_configured);
      }
      setApiError(false);
    } catch {
      setApiError(true);
    }
  }, []);

  useEffect(() => { void refreshData(); }, [refreshData]);
  useEffect(() => {
    void apiFetch(API + "/auth/me").then(async (response) => {
      if (response.ok) setAuthUser(await response.json());
      else if (response.status === 401) setAuthUser(null);
      setAuthChecked(true);
    }).catch(() => setAuthChecked(true));
  }, []);
  const submitAuth = async (event: FormEvent) => {
    event.preventDefault(); setAuthBusy(true); setAuthMessage("");
    try {
      const response = await fetch(API + "/auth/" + authMode, { method: "POST", credentials: "include", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email: authEmail, password: authPassword, ...(authMode === "register" ? { full_name: authName } : {}) }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Не удалось войти.");
      csrfToken = readCsrfToken(); setAuthUser(data); await refreshData();
    } catch (error) { setAuthMessage(error instanceof Error ? error.message : "Не удалось войти."); }
    finally { setAuthBusy(false); }
  };
  const logout = async () => { await apiFetch(API + "/auth/logout", { method: "POST" }); setAuthUser(null); setCandidates([]); setVacancies([]); };
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 3600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const openRoles = useMemo(() => vacancies.filter((vacancy) => vacancy.status === "open"), [vacancies]);
  const newestCandidates = useMemo(() => [...candidates].sort((a, b) => b.created_at.localeCompare(a.created_at)), [candidates]);

  const addMessage = (message: ChatMessage) => setMessages((current) => [...current, message]);
  const updateMessage = (id: string, update: (message: ChatMessage) => ChatMessage) => {
    setMessages((current) => current.map((message) => message.id === id ? update(message) : message));
  };

  const sendMessage = async (messageText: string) => {
    const text = messageText.trim();
    if (!text || chatPending) return;
    setChatOpen(true);
    setChatDraft("");
    setChatPending(true);
    const answerId = crypto.randomUUID();
    addMessage({ id: crypto.randomUUID(), role: "user", content: text });
    addMessage({ id: answerId, role: "assistant", content: "", streaming: true });
    try {
      const response = await apiFetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, conversation_id: conversationId }),
      });
      if (!response.ok || !response.body) {
        const problem = await response.json().catch(() => ({}));
        throw new Error(problem.detail ?? "Не удалось связаться с ассистентом.");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          const event = block.match(/^event:\s*(.+)$/m)?.[1]?.trim();
          const dataLine = block.split("\n").find((line) => line.startsWith("data:"));
          if (!event || !dataLine) continue;
          const data = JSON.parse(dataLine.slice(5).trim());
          if (event === "conversation") setConversationId(data.conversation_id);
          if (event === "token") updateMessage(answerId, (item) => ({ ...item, content: item.content + data.text }));
          if (event === "error") updateMessage(answerId, (item) => ({ ...item, content: data.message ?? "Произошла ошибка." }));
        }
        if (done) break;
      }
      updateMessage(answerId, (item) => ({ ...item, streaming: false }));
    } catch (error) {
      updateMessage(answerId, (item) => ({ ...item, content: error instanceof Error ? error.message : "Не удалось отправить сообщение.", streaming: false }));
    } finally {
      setChatPending(false);
    }
  };

  const submitUpload = async (event: FormEvent, audience: "hr" | "candidate") => {
    event.preventDefault();
    const file = audience === "candidate" ? portalFile : uploadFile;
    const candidateName = audience === "candidate" ? portalName : uploadName;
    if (!file) {
      setToast("Добавьте резюме в формате PDF или DOCX.");
      return;
    }
    if (!/\.(pdf|docx)$/i.test(file.name)) {
      setToast("Принимаются файлы PDF и DOCX.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setToast("Файл должен быть меньше 15 МБ.");
      return;
    }
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    if (candidateName.trim()) form.append("candidate_name", candidateName.trim());
    if (audience === "hr" && selectedVacancyId) form.append("vacancy_id", selectedVacancyId);
    form.append("source", audience === "candidate" ? "Портал кандидата" : uploadSource);
    try {
      const response = await apiFetch(`${API}${audience === "candidate" ? "/resumes/public" : "/resumes"}`, {
        method: "POST",
        headers: {},
        body: form,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "Резюме не удалось загрузить.");
      if (audience === "candidate") {
        setPortalSubmitted(true);
      } else {
        setUploadFile(null);
        setUploadName("");
        await refreshData();
        const indexed = result.sources?.some((source: CandidateDocument) => source.indexing_status === "indexed");
        if (!indexed) {
          setToast("Резюме сохранено, но индексирование не завершилось. Ассистент пока не видит его текст.");
          return;
        }
        setToast("Резюме добавлено. Ассистент уже может искать по нему.");
        const vacancy = vacancies.find((item) => item.id === selectedVacancyId);
        const vacancyContext = vacancy ? `\n\nВакансия для разбора: ${vacancy.title}. Требования: ${vacancy.requirements.map((item) => `${item.is_must_have ? "обязательно" : "желательно"}: ${item.text}`).join("; ") || "требования не указаны"}.` : "";
        void sendMessage(`Проверь нового кандидата ${result.full_name || "без имени"} (candidate_id: ${result.id}). Кратко опиши релевантный опыт по загруженному резюме, укажи источник и пробелы в данных. Не делай вывод о найме.${vacancyContext}`);
      }
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Ошибка загрузки.");
    } finally {
      setUploading(false);
    }
  };

  const submitLink = (event: FormEvent) => {
    event.preventDefault();
    const url = linkValue.trim();
    if (!/^https?:\/\//i.test(url)) {
      setToast("Вставьте полную ссылку, начиная с https://");
      return;
    }
    const prompt = [
      `Рекрутер получил ссылку на профиль кандидата: ${url}`,
      (() => {
        const vacancy = vacancies.find((item) => item.id === selectedVacancyId);
        return vacancy ? `Вакансия: ${vacancy.title}. Требования: ${vacancy.requirements.map((item) => `${item.is_must_have ? "обязательно" : "желательно"}: ${item.text}`).join("; ") || "не указаны"}.` : "";
      })(),
      linkContext.trim() ? `Контекст от рекрутера: ${linkContext.trim()}` : "",
      "Сначала проверь, есть ли уже данные этого кандидата в загруженных резюме. Не утверждай, что открыл ссылку: у тебя нет доступа к содержимому внешних страниц. Если данных нет, попроси загрузить PDF из hh/Telegram или вставить текст профиля, затем проанализируй материалы по требованиям вакансии.",
    ].filter(Boolean).join("\n\n");
    void sendMessage(prompt);
    setLinkValue("");
    setLinkContext("");
    setToast("Ссылка передана ассистенту как источник для проверки.");
  };

  const createVacancy = async (data: { title: string; department: string; description: string; requirements: string[] }) => {
    setCreatingVacancy(true);
    try {
      const response = await apiFetch(`${API}/vacancies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: data.title,
          department: data.department || null,
          description: data.description,
          requirements: data.requirements.map((text) => ({ text, is_must_have: true, weight: 3 })),
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "Не удалось создать вакансию.");
      setShowVacancyForm(false);
      setToast("Вакансия добавлена.");
      await refreshData();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Не удалось создать вакансию.");
    } finally {
      setCreatingVacancy(false);
    }
  };

  const openCandidate = async (candidate: Candidate) => {
    try {
      const response = await apiFetch(`${API}/candidates/${candidate.id}`);
      if (response.ok) setSelectedCandidate(await response.json());
      else setSelectedCandidate(candidate);
    } catch { setSelectedCandidate(candidate); }
  };

  if (!isCandidatePortal && (!authChecked || !authUser)) {
    return <div className="auth-shell"><form className="auth-card" onSubmit={submitAuth}><a className="brand auth-brand" href="/"><span className="brand-mark"><span /></span><span className="brand-name">линия<span>.</span></span></a><div className="eyebrow"><span className="eyebrow-line" /> РАБОЧЕЕ ПРОСТРАНСТВО</div><h1>{authMode === "login" ? "С возвращением." : "Создать аккаунт."}</h1><p>Войдите, чтобы работать с кандидатами и вакансиями.</p>{authMode === "register" && <label className="form-label"><span className="field-title">Ваше имя</span><input required minLength={2} value={authName} onChange={(event) => setAuthName(event.target.value)} placeholder="Имя и фамилия" /></label>}<label className="form-label"><span className="field-title">Рабочая почта</span><input type="email" required value={authEmail} onChange={(event) => setAuthEmail(event.target.value)} placeholder="name@company.ru" /></label><label className="form-label"><span className="field-title">Пароль</span><input type="password" required minLength={authMode === "register" ? 10 : 1} value={authPassword} onChange={(event) => setAuthPassword(event.target.value)} placeholder={authMode === "register" ? "Не менее 10 символов" : "Ваш пароль"} /></label>{authMessage && <div className="auth-error">{authMessage}</div>}<button className="button button-primary auth-submit" disabled={authBusy}>{authBusy ? "Подождите…" : authMode === "login" ? "Войти" : "Создать аккаунт"}</button><button className="button button-outline auth-submit" type="button" onClick={() => { window.location.href = API + "/auth/oauth/google"; }}>Продолжить с Google</button><button className="auth-switch" type="button" onClick={() => { setAuthMode(authMode === "login" ? "register" : "login"); setAuthMessage(""); }}>{authMode === "login" ? "Новый рекрутер? Создать аккаунт" : "Уже есть аккаунт? Войти"}</button><a className="auth-candidate-link" href="/candidate">Я кандидат — передать резюме</a></form></div>;
  }
  if (isCandidatePortal) {
    return <CandidatePortal name={portalName} setName={setPortalName} file={portalFile} setFile={setPortalFile} submitted={portalSubmitted} uploading={uploading} submit={(event) => void submitUpload(event, "candidate")} />;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Линия, главная">
          <span className="brand-mark"><span /></span><span className="brand-name">линия<span>.</span></span>
        </a>
        <div className="workspace-label">РАБОЧЕЕ ПРОСТРАНСТВО</div>
        <nav className="main-nav" aria-label="Основная навигация">
          <NavItem icon="home" label="Обзор" active={view === "overview"} onClick={() => setView("overview")} />
          <NavItem icon="people" label="Кандидаты" count={candidates.length} active={view === "candidates"} onClick={() => setView("candidates")} />
          <NavItem icon="briefcase" label="Вакансии" count={openRoles.length} active={view === "vacancies"} onClick={() => setView("vacancies")} />
          <NavItem icon="inbox" label="Входящие" active={view === "inbox"} onClick={() => setView("inbox")} />
        </nav>
        <div className="sidebar-foot">
          <a className="candidate-link" href="/candidate"><span className="portal-icon"><Glyph name="arrow" size={16} /></span><span><strong>Вы кандидат?</strong><small>Передать резюме</small></span></a>
          <div className="profile-chip"><div className="profile-avatar">{initials(authUser?.full_name ?? null)}</div><div className="profile-meta"><strong>{authUser?.full_name}</strong><span>{authUser?.role === "admin" ? "Администратор" : "Рекрутер"}</span></div><button className="auth-logout" onClick={() => void logout()}>Выйти</button></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="crumb"><span>Команда найма</span><Glyph name="chevron" size={14} /><strong>{viewLabels[view]}</strong></div>
          <div className="top-actions"><span className="today-label">{new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "long" }).format(new Date())}</span><button className="assistant-top-button" onClick={() => setChatOpen(true)}><Glyph name="sparkle" size={16} /> Ассистент <span className="live-dot" /></button></div>
        </header>

        {apiError && <div className="api-banner"><Glyph name="alert" size={16} /><span>Не удалось подключиться к API. Запустите backend на порту 8000.</span><button onClick={() => void refreshData()}>Повторить</button></div>}

        <div className="content-wrap">
          {view === "overview" && <Overview candidates={newestCandidates} vacancies={vacancies} openRoles={openRoles} onNavigate={setView} onCandidate={openCandidate} onAssistant={() => setChatOpen(true)} />}
          {view === "candidates" && <CandidatesPage candidates={newestCandidates} onCandidate={openCandidate} onUpload={() => { setView("inbox"); setActiveTab("file"); }} />}
          {view === "vacancies" && <VacanciesPage vacancies={vacancies} onCreate={() => setShowVacancyForm(true)} />}
          {view === "inbox" && <InboxPage activeTab={activeTab} setActiveTab={setActiveTab} uploadName={uploadName} setUploadName={setUploadName} uploadSource={uploadSource} setUploadSource={setUploadSource} uploadFile={uploadFile} setUploadFile={setUploadFile} uploading={uploading} submitUpload={(event) => void submitUpload(event, "hr")} linkValue={linkValue} setLinkValue={setLinkValue} linkContext={linkContext} setLinkContext={setLinkContext} submitLink={submitLink} vacancies={openRoles} selectedVacancyId={selectedVacancyId} setSelectedVacancyId={setSelectedVacancyId} />}
        </div>
        <footer className="page-footer"><span>Линия <span className="footer-dot">·</span> спокойный найм с опорой на данные</span><span className="connection"><i /> API {apiError ? "не подключён" : "подключён"}</span></footer>
      </main>

      {chatOpen && <AssistantPanel messages={messages} draft={chatDraft} setDraft={setChatDraft} pending={chatPending} configured={chatConfigured} onSend={() => void sendMessage(chatDraft)} onClose={() => setChatOpen(false)} />}
      {selectedCandidate && <CandidateDrawer candidate={selectedCandidate} onClose={() => setSelectedCandidate(null)} onAsk={() => { void sendMessage(`Проведи первичный разбор кандидата ${selectedCandidate.full_name || "без имени"} с ID ${selectedCandidate.id}. Используй его профиль и документы. Кратко укажи релевантный опыт, источники и вопросы, которые стоит уточнить.`); setSelectedCandidate(null); }} />}
      {showVacancyForm && <NewVacancyDialog onClose={() => setShowVacancyForm(false)} onCreate={createVacancy} busy={creatingVacancy} />}
      {toast && <div className="toast"><span className="toast-mark"><Glyph name="check" size={15} /></span>{toast}</div>}
    </div>
  );
}

const viewLabels: Record<View, string> = { overview: "Обзор", candidates: "Кандидаты", vacancies: "Вакансии", inbox: "Входящие" };

function NavItem({ icon, label, count, active, onClick }: { icon: string; label: string; count?: number; active: boolean; onClick: () => void }) {
  return <button className={`nav-item ${active ? "active" : ""}`} onClick={onClick}><Glyph name={icon} size={18} /><span>{label}</span>{count !== undefined && count > 0 && <small>{count}</small>}</button>;
}

function Overview({ candidates, vacancies, openRoles, onNavigate, onCandidate, onAssistant }: { candidates: Candidate[]; vacancies: Vacancy[]; openRoles: Vacancy[]; onNavigate: (view: View) => void; onCandidate: (candidate: Candidate) => void; onAssistant: () => void }) {
  return <>
    <section className="page-heading overview-heading">
      <div><div className="eyebrow"><span className="eyebrow-line" /> ПРОСТРАНСТВО КОМАНДЫ</div><h1>Работа с людьми<br /><span>без лишнего шума.</span></h1><p>Кандидаты, вакансии и аккуратная помощь ассистента — в одном месте.</p></div>
      <div className="heading-note"><span className="note-seal"><Glyph name="sparkle" size={18} /></span><span>Хороший найм<br />начинается с внимания.</span></div>
    </section>

    <section className="metric-row" aria-label="Сводка">
      <Metric label="Открытые вакансии" value={openRoles.length} caption="в работе" icon="briefcase" />
      <Metric label="Кандидаты" value={candidates.length} caption="в базе" icon="people" />
      <Metric label="Новые отклики" value={candidates.filter((candidate) => candidateStage(candidate) === "new").length} caption="нужен первый взгляд" icon="inbox" muted />
    </section>

    <div className="overview-grid">
      <section className="panel candidates-panel">
        <div className="panel-heading"><div><div className="eyebrow">ПОСЛЕДНИЕ ОТКЛИКИ</div><h2>Кандидаты</h2></div><button className="text-link" onClick={() => onNavigate("candidates")}>Все кандидаты <Glyph name="arrow" size={15} /></button></div>
        {candidates.length ? <CandidateTable candidates={candidates.slice(0, 5)} onCandidate={onCandidate} /> : <EmptyState title="Здесь появятся кандидаты" text="Добавьте PDF или DOCX либо передайте ассистенту ссылку на профиль." action="Добавить отклик" onClick={() => onNavigate("inbox")} />}
      </section>
      <div className="overview-aside">
        <section className="assistant-card">
          <div className="assistant-card-icon"><Glyph name="sparkle" size={20} /></div><span className="assistant-label">АССИСТЕНТ</span>
          <h2>Понять резюме<br />чуть быстрее.</h2><p>Найдёт опыт в документах и поможет сформулировать вопросы для интервью.</p>
          <button className="assistant-card-link" onClick={onAssistant}>Открыть чат <Glyph name="arrow" size={15} /></button>
          <div className="card-decoration decoration-one" /><div className="card-decoration decoration-two" />
        </section>
        <section className="roles-card"><div className="panel-heading compact"><div><div className="eyebrow">КОМАНДА ИЩЕТ</div><h3>Вакансии</h3></div><button className="icon-button" aria-label="Все вакансии" onClick={() => onNavigate("vacancies")}><Glyph name="arrow" size={16} /></button></div>{openRoles.length ? openRoles.slice(0, 3).map((role) => <button className="role-row" key={role.id} onClick={() => onNavigate("vacancies")}><span><strong>{role.title}</strong><small>{role.department || "Команда"}</small></span><Glyph name="chevron" size={15} /></button>) : <p className="empty-inline">Пока нет открытых вакансий.</p>}</section>
      </div>
    </div>
  </>;
}

function Metric({ label, value, caption, icon, muted = false }: { label: string; value: number; caption: string; icon: string; muted?: boolean }) {
  return <div className="metric"><span className={`metric-icon ${muted ? "soft" : ""}`}><Glyph name={icon} size={17} /></span><div className="metric-copy"><span>{label}</span><div><strong>{value.toString().padStart(2, "0")}</strong><small>{caption}</small></div></div></div>;
}

function CandidateTable({ candidates, onCandidate }: { candidates: Candidate[]; onCandidate: (candidate: Candidate) => void }) {
  return <div className="candidate-list"><div className="candidate-list-head"><span>КАНДИДАТ</span><span>ИСТОЧНИК</span><span>ДОБАВЛЕН</span><span>СТАТУС</span><span /></div>{candidates.map((candidate, index) => {
    const source = candidate.sources[0];
    return <button className="candidate-row" key={candidate.id} onClick={() => onCandidate(candidate)}><span className="candidate-person"><span className={`candidate-avatar avatar-${index % 4}`}>{initials(candidate.full_name)}</span><span><strong>{candidate.full_name || "Имя не указано"}</strong><small>{source?.filename || "Профиль кандидата"}</small></span></span><span className="source-label"><i className={source?.source.toLowerCase().includes("hh") ? "red-source" : ""} />{source?.source || "—"}</span><span className="row-date">{dateLabel(candidate.created_at)}</span><span><StatusPill status={candidateStage(candidate)} /></span><Glyph name="chevron" size={16} /></button>;
  })}</div>;
}

function StatusPill({ status }: { status: string }) { return <span className={`status-pill status-${status}`}>{statusLabels[status] || status}</span>; }

function EmptyState({ title, text, action, onClick }: { title: string; text: string; action: string; onClick: () => void }) {
  return <div className="empty-state"><span className="empty-illustration"><Glyph name="file" size={22} /></span><h3>{title}</h3><p>{text}</p><button className="button button-outline small" onClick={onClick}><Glyph name="plus" size={15} />{action}</button></div>;
}

function CandidatesPage({ candidates, onCandidate, onUpload }: { candidates: Candidate[]; onCandidate: (candidate: Candidate) => void; onUpload: () => void }) {
  return <><section className="page-heading compact-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> БАЗА КАНДИДАТОВ</div><h1>Люди, а не <span>строки.</span></h1><p>Материалы и источники собраны в карточках, чтобы важное не терялось.</p></div><button className="button button-primary" onClick={onUpload}><Glyph name="plus" size={17} /> Добавить кандидата</button></section><section className="panel full-panel"><div className="panel-heading"><div><div className="eyebrow">ВСЕ ОТКЛИКИ</div><h2>{candidates.length} {plural(candidates.length, "кандидат", "кандидата", "кандидатов")}</h2></div><span className="quiet-search"><Glyph name="search" size={15} /> Обновляется из базы</span></div>{candidates.length ? <CandidateTable candidates={candidates} onCandidate={onCandidate} /> : <EmptyState title="Пока тихо" text="Добавьте PDF или DOCX либо попросите ассистента проверить ссылку на профиль." action="Перейти во входящие" onClick={onUpload} />}</section></>;
}

function VacanciesPage({ vacancies, onCreate }: { vacancies: Vacancy[]; onCreate: () => void }) {
  return <>
    <section className="page-heading compact-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> РОЛИ И ТРЕБОВАНИЯ</div><h1>Работа <span>впереди.</span></h1><p>Открытые позиции и профиль поиска для каждой команды.</p></div><button className="button button-primary" onClick={onCreate}><Glyph name="plus" size={16} /> Добавить вакансию</button></section>
    <div className="vacancy-grid">{vacancies.map((vacancy, index) => <article className="vacancy-card" key={vacancy.id}><div className="vacancy-card-top"><span className={`vacancy-index tone-${index % 3}`}>0{index + 1}</span><StatusPill status={vacancy.status} /></div><h2>{vacancy.title}</h2><p className="vacancy-department">{vacancy.department || "Команда"}</p><p className="vacancy-description">{vacancy.description || "Описание роли пока не добавлено."}</p><div className="requirement-heading">ОЖИДАНИЯ <span>{vacancy.requirements.length}</span></div><div className="requirement-list">{vacancy.requirements.slice(0, 4).map((requirement) => <span key={requirement.id}><i className={requirement.is_must_have ? "must-have" : "nice-to-have"} />{requirement.text}</span>)}{vacancy.requirements.length > 4 && <small>+ ещё {vacancy.requirements.length - 4}</small>}</div></article>)}{!vacancies.length && <div className="panel vacancy-empty"><span className="empty-illustration"><Glyph name="briefcase" size={22} /></span><h3>Пока нет открытых ролей</h3><p>Добавьте позицию и требования — ассистент сможет учитывать их при разборе откликов.</p><button className="button button-outline small" onClick={onCreate}><Glyph name="plus" size={15} />Создать вакансию</button></div>}</div>
  </>;
}

function NewVacancyDialog({ onClose, onCreate, busy }: { onClose: () => void; onCreate: (data: { title: string; department: string; description: string; requirements: string[] }) => void; busy: boolean }) {
  const [title, setTitle] = useState("");
  const [department, setDepartment] = useState("");
  const [description, setDescription] = useState("");
  const [requirements, setRequirements] = useState("");
  return <div className="dialog-layer"><button className="drawer-scrim" onClick={onClose} aria-label="Закрыть форму" /><form className="vacancy-dialog" onSubmit={(event) => { event.preventDefault(); onCreate({ title: title.trim(), department: department.trim(), description: description.trim(), requirements: requirements.split("\n").map((line) => line.trim()).filter((line) => line.length >= 2) }); }}><header><div><div className="eyebrow">НОВАЯ ПОЗИЦИЯ</div><h2>Добавить вакансию</h2></div><button className="icon-button" type="button" onClick={onClose} aria-label="Закрыть"><Glyph name="close" size={18} /></button></header><label className="form-label"><span className="field-title">Название роли</span><input required minLength={2} maxLength={200} value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Например, Backend-разработчик" /></label><label className="form-label"><span className="field-title">Команда <em>необязательно</em></span><input maxLength={200} value={department} onChange={(event) => setDepartment(event.target.value)} placeholder="Платформа" /></label><label className="form-label"><span className="field-title">Коротко о роли <em>необязательно</em></span><textarea rows={3} maxLength={10000} value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Что предстоит делать и с кем работать" /></label><label className="form-label"><span className="field-title">Что важно для этой роли</span><textarea rows={4} value={requirements} onChange={(event) => setRequirements(event.target.value)} placeholder="Один пункт на строку\nОпыт Python от 3 лет\nУмение проектировать API" /></label><div className="dialog-actions"><button className="button button-outline" type="button" onClick={onClose}>Отмена</button><button className="button button-primary" type="submit" disabled={busy}>{busy ? <span className="spinner" /> : <Glyph name="check" size={15} />}{busy ? "Сохраняем" : "Создать вакансию"}</button></div></form></div>;
}

function InboxPage({ activeTab, setActiveTab, uploadName, setUploadName, uploadSource, setUploadSource, uploadFile, setUploadFile, uploading, submitUpload, linkValue, setLinkValue, linkContext, setLinkContext, submitLink, vacancies, selectedVacancyId, setSelectedVacancyId }: { activeTab: "file" | "link"; setActiveTab: (tab: "file" | "link") => void; uploadName: string; setUploadName: (value: string) => void; uploadSource: string; setUploadSource: (value: string) => void; uploadFile: File | null; setUploadFile: (file: File | null) => void; uploading: boolean; submitUpload: (event: FormEvent) => void; linkValue: string; setLinkValue: (value: string) => void; linkContext: string; setLinkContext: (value: string) => void; submitLink: (event: FormEvent) => void; vacancies: Vacancy[]; selectedVacancyId: string; setSelectedVacancyId: (value: string) => void }) {
  return <>
    <section className="page-heading compact-heading inbox-heading">
      <div><div className="eyebrow"><span className="eyebrow-line" /> ВХОДЯЩИЕ МАТЕРИАЛЫ</div><h1>Принять <span>отклик.</span></h1><p>Файл попадёт в базу, а ссылку можно передать ассистенту для следующего шага.</p></div>
      <span className="inbox-ornament"><Glyph name="inbox" size={30} /></span>
    </section>
    <div className="intake-layout">
      <section className="panel intake-panel">
        <div className="intake-tabs">
          <button className={activeTab === "file" ? "selected" : ""} onClick={() => setActiveTab("file")}><Glyph name="file" size={16} /> Резюме PDF/DOCX</button>
          <button className={activeTab === "link" ? "selected" : ""} onClick={() => setActiveTab("link")}><Glyph name="link" size={16} /> Ссылка</button>
        </div>
        {activeTab === "file" ? <form onSubmit={submitUpload} className="intake-form">
          <label className="form-label"><span className="field-title">Имя кандидата <em>необязательно</em></span><input value={uploadName} onChange={(event) => setUploadName(event.target.value)} placeholder="Например, Анна Петрова" /></label>
          <label className="form-label"><span className="field-title">Сопоставить с вакансией <em>необязательно</em></span><select value={selectedVacancyId} onChange={(event) => setSelectedVacancyId(event.target.value)}><option value="">Без вакансии</option>{vacancies.map((vacancy) => <option value={vacancy.id} key={vacancy.id}>{vacancy.title}</option>)}</select></label>
          <label className="form-label"><span className="field-title">Откуда пришёл отклик</span><select value={uploadSource} onChange={(event) => setUploadSource(event.target.value)}><option>Почта отдела кадров</option><option>Telegram</option><option>hh.ru</option><option>Рекомендация</option><option>Сайт компании</option></select></label>
          <FilePicker file={uploadFile} onChange={setUploadFile} />
          <div className="intake-submit-row"><span><Glyph name="check" size={14} /> PDF или DOCX до 15 МБ · текст будет проиндексирован</span><button className="button button-primary" type="submit" disabled={uploading}>{uploading ? <span className="spinner" /> : <Glyph name="upload" size={16} />}{uploading ? "Загружаем" : "Добавить резюме"}</button></div>
        </form> : <form onSubmit={submitLink} className="intake-form link-form">
          <label className="form-label"><span className="field-title">Ссылка на профиль</span><input type="url" required value={linkValue} onChange={(event) => setLinkValue(event.target.value)} placeholder="https://hh.ru/resume/... или t.me/..." /></label>
          <label className="form-label"><span className="field-title">Вакансия для разбора <em>необязательно</em></span><select value={selectedVacancyId} onChange={(event) => setSelectedVacancyId(event.target.value)}><option value="">Без вакансии</option>{vacancies.map((vacancy) => <option value={vacancy.id} key={vacancy.id}>{vacancy.title}</option>)}</select></label>
          <label className="form-label"><span className="field-title">Что важно знать <em>необязательно</em></span><textarea rows={4} value={linkContext} onChange={(event) => setLinkContext(event.target.value)} placeholder="Вакансия, источник, что проверить в первую очередь" /></label>
          <div className="link-note"><span className="note-icon"><Glyph name="alert" size={15} /></span><span><strong>Ассистент не открывает внешние страницы.</strong> Ссылка и заметка попадут в историю чата; для разбора содержания приложите PDF или вставьте текст профиля.</span></div>
          <div className="intake-submit-row"><span><Glyph name="link" size={14} /> hh.ru · Telegram · другие источники</span><button className="button button-primary" type="submit"><Glyph name="sparkle" size={16} /> Передать ассистенту</button></div>
        </form>}
      </section>
      <aside className="intake-aside">
        <div className="eyebrow">КАК ЭТО РАБОТАЕТ</div>
        <div className="flow-step"><span className="flow-number">01</span><span><strong>Рекрутер получает отклик</strong><small>Файл из почты или экспорт профиля из hh / Telegram.</small></span></div><div className="flow-rule" />
        <div className="flow-step"><span className="flow-number">02</span><span><strong>Материал попадает в контекст</strong><small>PDF сохраняется в карточке; ссылка передаётся в чат с заметкой.</small></span></div><div className="flow-rule" />
        <div className="flow-step"><span className="flow-number">03</span><span><strong>Ассистент делает первичный разбор</strong><small>Опирается на загруженный текст, ссылается на источник и отмечает пробелы.</small></span></div>
        <div className="aside-footnote">Решение о кандидате всегда остаётся за командой.</div>
      </aside>
    </div>
  </>;
}

function FilePicker({ file, onChange }: { file: File | null; onChange: (file: File | null) => void }) {
  const handleChange = (event: ChangeEvent<HTMLInputElement>) => onChange(event.target.files?.[0] ?? null);
  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    onChange(event.dataTransfer.files?.[0] ?? null);
  };
  return <label className={`file-picker ${file ? "has-file" : ""}`} onDragOver={(event) => event.preventDefault()} onDrop={handleDrop}>
    <input type="file" accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" onChange={handleChange} />
    <span className="file-picker-icon"><Glyph name={file ? "check" : "upload"} size={19} /></span>
    <span className="file-picker-copy"><strong>{file ? file.name : "Перетащите PDF или DOCX сюда или выберите файл"}</strong><small>{file ? `${(file.size / (1024 * 1024)).toFixed(1)} МБ · выберите снова, чтобы заменить` : "PDF или DOCX, до 15 МБ"}</small></span>
  </label>;
}

function AssistantPanel({ messages, draft, setDraft, pending, configured, onSend, onClose }: { messages: ChatMessage[]; draft: string; setDraft: (value: string) => void; pending: boolean; configured: boolean | null; onSend: () => void; onClose: () => void }) {
  const listRef = useCallback((node: HTMLDivElement | null) => { if (node) node.scrollTop = node.scrollHeight; }, [messages.length, messages[messages.length - 1]?.content]);
  const suggestions = ["Какие кандидаты подходят на эту роль?", "Сравни опыт двух кандидатов", "Какие вопросы задать на интервью?"];
  return <><button className="drawer-scrim" onClick={onClose} aria-label="Закрыть чат" /><aside className="assistant-panel"><header className="chat-header"><div className="chat-brand"><span className="chat-brand-icon"><Glyph name="sparkle" size={17} /></span><span><strong>Ассистент</strong><small><i /> Помощник команды</small></span></div><button className="icon-button" onClick={onClose} aria-label="Закрыть"><Glyph name="close" size={18} /></button></header><div className="chat-context"><span>КОНТЕКСТ</span><p>Загруженные резюме и открытые вакансии</p></div><div className="chat-messages" ref={listRef}>{!messages.length && <div className="chat-welcome"><span className="welcome-sparkle"><Glyph name="sparkle" size={23} /></span><div className="eyebrow">РАБОЧИЙ ПОМОЩНИК</div><h2>С чего начнём?</h2><p>Я помогу найти опыт в резюме и подготовить вопросы для интервью. Решение остаётся за вами.</p><div className="suggestion-list">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => { setDraft(suggestion); }}>{suggestion}<Glyph name="arrow" size={14} /></button>)}</div></div>}{messages.map((message) => <div className={`message ${message.role}`} key={message.id}><span className="message-avatar">{message.role === "assistant" ? <Glyph name="sparkle" size={14} /> : "Вы"}</span><div className="message-content">{message.role === "assistant" ? <div className="markdown-content"><ReactMarkdown remarkPlugins={[remarkGfm]}>{normalizeAssistantMarkdown(message.content)}</ReactMarkdown></div> : message.content}{message.streaming && !message.content && <span className="typing"><i /><i /><i /></span>}{message.streaming && message.content && <span className="stream-cursor" />}</div></div>)}{!configured && configured !== null && <div className="chat-config-note">Для чата добавьте OPENAI_API_KEY в файл .env.</div>}</div><form className="chat-compose" onSubmit={(event) => { event.preventDefault(); onSend(); }}><textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Спросите о кандидатах или вакансии…" rows={3} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); onSend(); } }} /><div className="compose-bottom"><span>Enter — отправить · Shift + Enter — новая строка</span><button className="send-button" type="submit" disabled={!draft.trim() || pending} aria-label="Отправить"><Glyph name="send" size={17} /></button></div></form></aside></>;
}

function CandidateDrawer({ candidate, onClose, onAsk }: { candidate: Candidate; onClose: () => void; onAsk: () => void }) {
  return <><button className="drawer-scrim candidate-scrim" onClick={onClose} aria-label="Закрыть карточку" /><aside className="candidate-drawer"><header className="candidate-drawer-head"><span className="eyebrow">КАРТОЧКА КАНДИДАТА</span><button className="icon-button" onClick={onClose} aria-label="Закрыть"><Glyph name="close" size={18} /></button></header><div className="candidate-profile"><div className="large-avatar">{initials(candidate.full_name)}</div><h2>{candidate.full_name || "Имя не указано"}</h2><StatusPill status={candidateStage(candidate)} /><p>{candidate.email || candidate.phone || "Контактные данные не добавлены"}</p></div><div className="profile-divider" /><div className="eyebrow">ОТКЛИКИ</div><div className="source-documents">{candidate.applications.length ? candidate.applications.map((application) => <div className="source-document" key={application.id}><span><Glyph name="briefcase" size={17} /></span><div><strong>{application.vacancy_title}</strong><small>{statusLabels[application.stage] || application.stage}</small></div></div>) : <p className="empty-inline">Кандидат пока не привязан к вакансии.</p>}</div><div className="eyebrow">ИСТОЧНИКИ</div><div className="source-documents">{candidate.sources.length ? candidate.sources.map((source) => <div className="source-document" key={source.id}><span><Glyph name="file" size={17} /></span><div><strong>{source.filename}</strong><small>{source.source} · {dateLabel(source.uploaded_at)}</small></div><i className={source.indexing_status === "indexed" ? "indexed" : "pending-index"} /></div>) : <p className="empty-inline">Документы ещё не добавлены.</p>}</div><div className="candidate-drawer-actions"><button className="button button-primary" onClick={onAsk}><Glyph name="sparkle" size={16} /> Попросить ассистента разобрать</button><span>Ассистент использует только доступные материалы и назовёт источники.</span></div><div className="drawer-date">В базе с {dateLabel(candidate.created_at)}</div></aside></>;
}

function CandidatePortal({ name, setName, file, setFile, submitted, uploading, submit }: { name: string; setName: (value: string) => void; file: File | null; setFile: (file: File | null) => void; submitted: boolean; uploading: boolean; submit: (event: FormEvent) => void }) {
  return <div className="portal-shell">
    <header className="portal-header"><a className="brand" href="/"><span className="brand-mark"><span /></span><span className="brand-name">линия<span>.</span></span></a><a className="portal-back" href="/">Для команды найма <Glyph name="arrow" size={15} /></a></header>
    <main className="portal-main">
      <div className="portal-kicker"><span className="eyebrow-line" /> ПОРТАЛ КАНДИДАТА</div>
      {submitted ? <section className="portal-success"><span className="success-icon"><Glyph name="check" size={23} /></span><div className="eyebrow">РЕЗЮМЕ ПОЛУЧЕНО</div><h1>Спасибо, <span>что откликнулись.</span></h1><p>Файл сохранён. Команда найма сможет посмотреть его и вернуться к вам, если опыт совпадёт с открытой ролью.</p><a href="/" className="button button-outline">Вернуться на главную <Glyph name="arrow" size={15} /></a></section> :
        <div className="portal-grid">
          <section className="portal-copy"><h1>Расскажите<br /><span>о своём опыте.</span></h1><p>Загрузите резюме, чтобы команда могла внимательно ознакомиться с вашим опытом. Ваш документ будет прикреплён к карточке кандидата.</p><div className="portal-promise"><span><Glyph name="check" size={15} /></span><span><strong>Только по делу</strong><small>Никаких длинных форм — достаточно PDF или DOCX.</small></span></div><div className="portal-promise"><span><Glyph name="check" size={15} /></span><span><strong>С заботой о контексте</strong><small>Ассистент помогает команде читать документы, не принимает решения о найме.</small></span></div></section>
          <form className="portal-form" onSubmit={submit}><div className="eyebrow">ВАШ ОТКЛИК</div><label className="form-label"><span className="field-title">Как к вам обращаться</span><input value={name} onChange={(event) => setName(event.target.value)} placeholder="Имя и фамилия" /></label><div className="form-label"><span className="field-title">Резюме в PDF или DOCX</span></div><FilePicker file={file} onChange={setFile} /><button className="button button-primary portal-submit" type="submit" disabled={uploading}>{uploading ? <span className="spinner" /> : <Glyph name="upload" size={16} />}{uploading ? "Отправляем" : "Передать резюме"}</button><p className="privacy-note">Отправляя файл, вы передаёте его команде найма для рассмотрения отклика.</p></form>
        </div>}
    </main>
    <footer className="portal-footer"><span>Линия <span className="footer-dot">·</span> рабочее пространство найма</span><span>Вопросы? Ответьте на письмо с вакансией.</span></footer>
  </div>;
}

function plural(count: number, one: string, few: string, many: string) {
  const ten = count % 10;
  const hundred = count % 100;
  if (ten === 1 && hundred !== 11) return one;
  if (ten >= 2 && ten <= 4 && (hundred < 12 || hundred > 14)) return few;
  return many;
}
