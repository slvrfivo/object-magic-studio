import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';
import { useStudioStore } from './store/studioStore';

// 개발 모드 전용: 브라우저 자동화 테스트(미리보기 ↔ 내보내기 일치 검증)에서 상태를 읽기 위한 훅
if (import.meta.env.DEV) {
  (window as unknown as { __oms: typeof useStudioStore }).__oms = useStudioStore;
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
