venvPy = fullfile(pwd, '.venv', 'bin', 'python');
pe = pyenv;
if pe.Status == "Loaded" && pe.Executable ~= venvPy
    error("MATLAB already loaded a different Python: %s", pe.Executable);
elseif pe.Status ~= "Loaded"
    pyenv('Version', venvPy);
end

