import React, { useState } from 'react';
import { Search, AlertTriangle, ShieldCheck, Database, Info, Loader2 } from 'lucide-react';

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('http://localhost:8000/api/v1/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, top_k: 2 }),
      });

      const data = await res.json();

      if (!res.ok) {
        // Handle 422 Guardrail Failures or 502/500 Server Errors
        throw new Error(data.detail?.error || data.detail?.message || "An unexpected error occurred");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center space-x-3 text-blue-600 mb-4">
            <ShieldCheck size={40} />
            <h1 className="text-3xl font-bold">Deriv RAG Assistant</h1>
          </div>
          <p className="text-slate-500">Ask verified technical questions about Deriv synthetic indices, CFDs, and API payloads.</p>
        </div>

        {/* Search Bar */}
        <form onSubmit={handleSubmit} className="relative shadow-sm rounded-xl overflow-hidden bg-white border border-slate-200 focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500">
          <div className="flex items-center px-4 py-3">
            <Search className="text-slate-400 mr-3" size={20} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. What happens when margin hits 50% on CFDs?"
              className="flex-1 outline-none text-lg bg-transparent"
              disabled={loading}
            />
            <button 
              type="submit" 
              disabled={loading || !query.trim()}
              className="ml-4 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center"
            >
              {loading ? <Loader2 className="animate-spin" size={20} /> : 'Ask Agent'}
            </button>
          </div>
        </form>

        {/* Error State (Guardrails & API Drops) */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-5 flex items-start space-x-4">
            <AlertTriangle className="text-red-500 mt-1 shrink-0" size={24} />
            <div>
              <h3 className="text-red-800 font-semibold text-lg">Query Failed</h3>
              <p className="text-red-600 mt-1">{error}</p>
              <p className="text-red-500 text-sm mt-2 font-medium">Safe Fallback: Please refer directly to the Deriv official documentation.</p>
            </div>
          </div>
        )}

        {/* Success Results */}
        {result && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            
            {/* The Answer */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-4">
              <h2 className="text-xl font-semibold flex items-center">
                <Info className="text-blue-500 mr-2" size={20} /> 
                Generated Answer
              </h2>
              <p className="text-slate-700 leading-relaxed text-lg">
                {result.structured_output.answer}
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Extracted Parameters */}
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="text-lg font-semibold mb-4 border-b pb-2">Extracted Specifications</h3>
                {Object.keys(result.structured_output.exact_parameters).length > 0 ? (
                  <ul className="space-y-3">
                    {Object.entries(result.structured_output.exact_parameters).map(([key, val]) => (
                      <li key={key} className="flex flex-col justify-between p-3 bg-slate-50 rounded-lg border border-slate-100">
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-mono text-sm text-slate-500">{key}</span>
                          <span className="font-bold text-slate-800">{val}</span>
                        </div>
                        {/* Show chunk source attribution for this specific parameter */}
                        {result.parameter_sources[key] && result.parameter_sources[key].length > 0 && (
                          <div className="text-xs text-blue-600 mt-1 flex items-center">
                            <Database size={12} className="mr-1" />
                            Source: {result.parameter_sources[key][0]}
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-slate-400 italic">No exact numeric specs found in this answer.</p>
                )}
              </div>

              {/* Risk Warning & Sources */}
              <div className="space-y-6">
                <div className="bg-orange-50 rounded-xl border border-orange-200 p-6">
                  <h3 className="text-lg font-semibold text-orange-800 mb-2 flex items-center">
                    <AlertTriangle className="mr-2" size={18} /> Required Risk Warning
                  </h3>
                  <p className="text-orange-700 text-sm leading-relaxed">
                    {result.structured_output.risk_warning}
                  </p>
                </div>

                <div className="bg-slate-800 rounded-xl p-6 text-slate-200">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
                    Retrieved Documents
                  </h3>
                  <ul className="space-y-2">
                    {result.sources.map((src, i) => (
                      <li key={i} className="flex items-center text-sm">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mr-2" />
                        {src}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}