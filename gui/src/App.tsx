import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { ToastProvider } from './components/Toast';

// Bundle Optimization (bundle-dynamic-imports): Lazy load heavy route pages
const ProjectView = lazy(() => import('./pages/ProjectView').then(m => ({ default: m.ProjectView })));
const ScriptStudio = lazy(() => import('./pages/ScriptStudio').then(m => ({ default: m.ScriptStudio })));
const Settings = lazy(() => import('./pages/Settings').then(m => ({ default: m.Settings })));

export const App: React.FC = () => {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Layout>
          <Suspense fallback={
            <div style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px', padding: '40px' }}>
              <span className="spinner" /> Loading view...
            </div>
          }>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/project/:name" element={<ProjectView />} />
              <Route path="/project/:name/script" element={<ScriptStudio />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Suspense>
        </Layout>
      </BrowserRouter>
    </ToastProvider>
  );
};

export default App;
