import React, { useEffect, useState, useCallback } from 'react';

const FileTreeItem = ({ item, onSelect, selectedPath }) => {
    const [isOpen, setIsOpen] = useState(true);

    if (item.type === 'file' || item.content !== undefined) {
        const isSelected = selectedPath === item.path;
        return (
            <div
                className={`tree-item tree-file ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelect(item)}
            >
                📄 {item.name}
            </div>
        );
    }

    // It's a folder
    return (
        <div>
            <div
                className="tree-item tree-folder"
                onClick={() => setIsOpen(!isOpen)}
            >
                {isOpen ? '📂' : '📁'} {item.name}
            </div>
            {isOpen && (
                <div style={{ paddingLeft: '15px' }}>
                    {item.children.map((child, idx) => (
                        <FileTreeItem
                            key={idx}
                            item={{
                                ...child,
                                path: child.path || (item.path ? `${item.path}/${child.name}` : child.name),
                            }}
                            onSelect={onSelect}
                            selectedPath={selectedPath}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

const FileExplorer = ({ tree, files, onLoadFile, onSaveFile, projectName, isTreeLoading }) => {
    const [selectedPath, setSelectedPath] = useState(null);
    const [editorValue, setEditorValue] = useState("");
    const [isDirty, setIsDirty] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [isLoadingFile, setIsLoadingFile] = useState(false);
    const [loadError, setLoadError] = useState(null);

    // Load (or re-load) a file by path from the server
    const loadFileContent = useCallback(async (path) => {
        if (!path || !onLoadFile) return;
        setIsLoadingFile(true);
        setLoadError(null);
        try {
            const content = await onLoadFile(path);
            setEditorValue(content || "");
            setIsDirty(false);
        } catch (e) {
            setLoadError(e?.message || "Failed to load file");
        } finally {
            setIsLoadingFile(false);
        }
    }, [onLoadFile]);

    // When the selected file's cached content changes (including being evicted → undefined),
    // re-sync the editor. If evicted (undefined), force a fresh fetch from the server.
    useEffect(() => {
        if (!selectedPath) return;

        const cached = files?.[selectedPath];
        if (cached !== undefined) {
            // Fresh content available in cache — update editor without marking dirty
            setEditorValue(cached);
            setIsDirty(false);
            setLoadError(null);
        } else {
            // Content was evicted (e.g. after a code update) — re-fetch from disk
            loadFileContent(selectedPath);
        }
    }, [files, selectedPath, loadFileContent]);

    const handleSelect = async (item) => {
        if (!item) return;
        const path = item.path;
        if (!path) return;
        if (item.type !== 'file' && item.content === undefined) return;

        setSelectedPath(path);
        setLoadError(null);

        // Always fetch fresh from server when clicking a file
        // (avoids showing stale cache after code updates)
        await loadFileContent(path);
    };

    const handleSave = async () => {
        if (!selectedPath || !onSaveFile) return;
        setIsSaving(true);
        try {
            await onSaveFile(selectedPath, editorValue);
            setIsDirty(false);
        } finally {
            setIsSaving(false);
        }
    };

    if (isTreeLoading) {
        return (
            <div style={{ padding: 40, textAlign: 'center', color: '#666', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                <span style={{ display: 'inline-block', width: 24, height: 24, border: '2px solid #e2e8f0', borderTopColor: '#3b82f6', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
                <span>Loading file tree...</span>
            </div>
        );
    }

    if (!tree) {
        return (
            <div style={{ padding: 20, textAlign: 'center', color: '#666' }}>
                No files generated yet.
            </div>
        );
    }

    return (
        <div className="file-explorer">
            <div className="file-tree">
                {tree.map((item, idx) => (
                    <FileTreeItem key={idx} item={item} onSelect={handleSelect} selectedPath={selectedPath} />
                ))}
            </div>
            <div className="code-viewer">
                <div className="code-header">
                    <div className="code-header-row">
                        <div className="code-header-title">
                            {selectedPath ? selectedPath : "Select a file to view"}
                            {isDirty ? <span className="dirty-indicator">•</span> : null}
                            {isLoadingFile ? <span style={{ marginLeft: 8, fontSize: 11, color: '#3b82f6' }}>Loading...</span> : null}
                        </div>
                        <div className="code-header-actions">
                            <button
                                className="save-button"
                                onClick={handleSave}
                                disabled={!selectedPath || !isDirty || isSaving || !projectName}
                                title={!projectName ? "Generate a project first" : "Save file"}
                            >
                                {isSaving ? "Saving..." : "Save"}
                            </button>
                        </div>
                    </div>
                </div>
                <div className="code-content">
                    {loadError ? (
                        <div style={{ color: '#b00020' }}>{loadError}</div>
                    ) : (
                        <textarea
                            className="code-editor"
                            value={editorValue}
                            onChange={(e) => {
                                setEditorValue(e.target.value);
                                setIsDirty(true);
                            }}
                            spellCheck={false}
                            placeholder={selectedPath ? (isLoadingFile ? "Loading..." : "Edit code...") : "Select a file from the tree"}
                            disabled={isLoadingFile}
                        />
                    )}
                </div>
            </div>
        </div>
    );
};

export default FileExplorer;
