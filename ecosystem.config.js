module.exports = {
    apps: [{
        "cwd": "/Users/portlet/Workspace/demo/knowledge_gpt/knowledge_gpt",
        "error_file": "~/.pm2/logs/knowledge_gpt-error.log",
        "out_file": "~/.pm2/logs/knowledge_gpt-out.log",
        "exec_mode": "fork",
        "instances": 1,
        // "interpreter": "/Users/portlet/Workspace/demo/knowledge_gpt/.venv/bin/python",
        "kill_timeout": 10000,
        "log_data_format": "YYYY-MM-DD HH:mm",
        "max_memory_restart": "2G",
        "merge_logs": true,
        "name": "knowledge_gpt",
        "script": "python",
        "args": "-m streamlit run hello.py --server.port 8501 --server.baseUrlPath /chat/ --server.enableCORS false --server.enableXsrfProtection false --server.headless=true",
        // "args": [
        //     "-m",
        //     "streamlit",
        //     "run",
        //     "main.py",
        //     "--server.port=8501",
        //     "--server.baseUrlPath=/",
        //     "--server.enableCORS=false",
        //     "--server.enableXsrfProtection=false",
        //     "--server.headless=true"
        // ],
        "time": true,
        "wait_ready": true
    }]
}