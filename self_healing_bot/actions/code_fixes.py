"""Code repair actions for common ML pipeline issues."""

import re
from typing import Dict, Any, List, Optional
import logging

from .base import BaseAction, ActionResult
from ..core.context import Context

logger = logging.getLogger(__name__)


class CodeFixAction(BaseAction):
    """Apply common code fixes for ML pipeline issues."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.fix_patterns = self.config.get("fix_patterns", self._get_default_patterns())
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["import_error", "syntax_error", "type_error", "gpu_oom"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute code fixes based on issue type."""
        issue_type = issue_data.get("type", "")
        
        if not self.can_handle(issue_type):
            return self.create_result(
                success=False,
                message=f"Cannot handle issue type: {issue_type}"
            )
        
        try:
            if issue_type == "import_error":
                return await self._fix_import_error(context, issue_data)
            elif issue_type == "gpu_oom":
                return await self._fix_gpu_oom(context, issue_data)
            elif issue_type == "syntax_error":
                return await self._fix_syntax_error(context, issue_data)
            elif issue_type == "type_error":
                return await self._fix_type_error(context, issue_data)
            else:
                return self.create_result(
                    success=False,
                    message=f"No specific fix available for {issue_type}"
                )
                
        except Exception as e:
            logger.exception(f"Error applying code fix: {e}")
            return self.create_result(
                success=False,
                message=f"Code fix failed: {str(e)}"
            )
    
    async def _fix_import_error(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix common import errors."""
        missing_module = issue_data.get("missing_module", "")
        affected_file = issue_data.get("affected_file", "")
        
        if not missing_module or not affected_file:
            return self.create_result(
                success=False,
                message="Missing module name or affected file information"
            )
        
        # Common import fixes
        import_fixes = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "dotenv": "python-dotenv"
        }
        
        # Try to fix the import
        file_content = context.read_file(affected_file)
        
        # Add missing import
        if missing_module in import_fixes:
            # Update requirements if needed
            requirements_content = context.read_file("requirements.txt")
            if import_fixes[missing_module] not in requirements_content:
                new_requirements = requirements_content + f"\n{import_fixes[missing_module]}\n"
                context.write_file("requirements.txt", new_requirements)
            
            # Fix import statement
            fixed_content = self._add_import_statement(file_content, missing_module)
            context.write_file(affected_file, fixed_content)
            
            return self.create_result(
                success=True,
                message=f"Fixed import error for {missing_module} in {affected_file}",
                data={"fixed_import": missing_module, "file": affected_file}
            )
        
        return self.create_result(
            success=False,
            message=f"No automatic fix available for module: {missing_module}"
        )
    
    async def _fix_gpu_oom(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix GPU out-of-memory errors."""
        affected_files = issue_data.get("affected_files", ["train.py"])
        fixes_applied = []
        
        for file_path in affected_files:
            try:
                content = context.read_file(file_path)
                modified = False
                
                # Reduce batch size
                batch_size_pattern = r'batch_size\s*=\s*(\d+)'
                match = re.search(batch_size_pattern, content)
                if match:
                    old_size = int(match.group(1))
                    new_size = max(1, old_size // 2)
                    content = re.sub(batch_size_pattern, f'batch_size = {new_size}', content)
                    fixes_applied.append(f"Reduced batch size from {old_size} to {new_size}")
                    modified = True
                
                # Enable gradient checkpointing
                if "model.gradient_checkpointing_enable" not in content and "torch" in content:
                    # Find model definition and add gradient checkpointing
                    model_pattern = r'(model\s*=\s*.*?\n)'
                    if re.search(model_pattern, content):
                        content = re.sub(
                            model_pattern,
                            r'\1model.gradient_checkpointing_enable()\n',
                            content
                        )
                        fixes_applied.append("Enabled gradient checkpointing")
                        modified = True
                
                # Add memory cleanup
                if "torch.cuda.empty_cache()" not in content and "torch" in content:
                    # Add memory cleanup after training steps
                    content += "\n# GPU memory cleanup\ntorch.cuda.empty_cache()\n"
                    fixes_applied.append("Added GPU memory cleanup")
                    modified = True
                
                if modified:
                    context.write_file(file_path, content)
                    
            except Exception as e:
                logger.warning(f"Could not fix GPU OOM in {file_path}: {e}")
        
        if fixes_applied:
            return self.create_result(
                success=True,
                message=f"Applied GPU OOM fixes: {', '.join(fixes_applied)}",
                data={"fixes": fixes_applied}
            )
        else:
            return self.create_result(
                success=False,
                message="No GPU OOM fixes could be applied"
            )
    
    async def _fix_syntax_error(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix common syntax errors."""
        affected_file = issue_data.get("affected_file", "")
        error_line = issue_data.get("error_line", 0)
        
        if not affected_file:
            return self.create_result(
                success=False,
                message="Missing affected file information"
            )
        
        try:
            content = context.read_file(affected_file)
            lines = content.split('\n')
            
            if error_line > 0 and error_line <= len(lines):
                line = lines[error_line - 1]
                
                # Common syntax fixes
                fixed_line = line
                fixes_applied = []
                
                # Missing colon
                if re.match(r'^\s*(if|for|while|def|class|try|except|else|elif)', line) and not line.rstrip().endswith(':'):
                    fixed_line = line.rstrip() + ':'
                    fixes_applied.append("Added missing colon")
                
                # Mismatched parentheses/brackets
                if line.count('(') != line.count(')'):
                    # Simple fix: add missing closing parenthesis
                    if line.count('(') > line.count(')'):
                        fixed_line = line.rstrip() + ')'
                        fixes_applied.append("Added missing closing parenthesis")
                
                if fixes_applied:
                    lines[error_line - 1] = fixed_line
                    context.write_file(affected_file, '\n'.join(lines))
                    
                    return self.create_result(
                        success=True,
                        message=f"Fixed syntax error on line {error_line}: {', '.join(fixes_applied)}",
                        data={"fixes": fixes_applied, "line": error_line}
                    )
            
            return self.create_result(
                success=False,
                message="Could not automatically fix syntax error"
            )
            
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Failed to fix syntax error: {str(e)}"
            )
    
    async def _fix_type_error(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix common type errors."""
        # This is a placeholder for type error fixes
        return self.create_result(
            success=False,
            message="Type error fixes not yet implemented"
        )
    
    def _add_import_statement(self, content: str, module: str) -> str:
        """Add import statement to file content."""
        lines = content.split('\n')
        
        # Find the right place to insert the import (after existing imports)
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_idx = i + 1
            elif line.strip() == '':
                continue
            else:
                break
        
        # Insert the import
        import_line = f"import {module}"
        lines.insert(insert_idx, import_line)
        
        return '\n'.join(lines)
    
    def _get_default_patterns(self) -> Dict[str, List[str]]:
        """Get default fix patterns."""
        return {
            "import_fixes": [
                r"No module named '(\w+)'",
                r"ImportError: (.+)",
                r"ModuleNotFoundError: No module named '(.+)'"
            ],
            "gpu_oom": [
                r"CUDA out of memory",
                r"RuntimeError: CUDA out of memory",
                r"GPU memory.*insufficient"
            ],
            "syntax_errors": [
                r"SyntaxError: (.+)",
                r"IndentationError: (.+)",
                r"TabError: (.+)"
            ]
        }


class DependencyFixAction(BaseAction):
    """Fix dependency-related issues."""
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in ["dependency_error", "version_conflict"]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix dependency issues."""
        try:
            # Read current requirements
            requirements_content = context.read_file("requirements.txt")
            
            # Common dependency fixes
            dependency_fixes = {
                "tensorflow": "tensorflow>=2.8.0",
                "torch": "torch>=1.12.0",
                "scikit-learn": "scikit-learn>=1.0.0",
                "pandas": "pandas>=1.3.0",
                "numpy": "numpy>=1.21.0"
            }
            
            missing_deps = issue_data.get("missing_dependencies", [])
            fixes_applied = []
            
            for dep in missing_deps:
                if dep in dependency_fixes:
                    if dep not in requirements_content:
                        requirements_content += f"\n{dependency_fixes[dep]}\n"
                        fixes_applied.append(dep)
            
            if fixes_applied:
                context.write_file("requirements.txt", requirements_content)
                return self.create_result(
                    success=True,
                    message=f"Added missing dependencies: {', '.join(fixes_applied)}",
                    data={"added_dependencies": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No dependency fixes applied"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Dependency fix failed: {str(e)}"
            )


class ImportFixAction(BaseAction):
    """Fix import-related issues specifically."""
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type == "import_error"
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix import errors with more sophisticated logic."""
        try:
            missing_imports = issue_data.get("missing_imports", [])
            affected_files = issue_data.get("affected_files", [])
            
            fixes_applied = []
            
            for file_path in affected_files:
                content = context.read_file(file_path)
                
                for missing_import in missing_imports:
                    # Check if import is already present
                    if f"import {missing_import}" not in content and f"from {missing_import}" not in content:
                        # Add the import at the top
                        lines = content.split('\n')
                        
                        # Find insertion point (after docstring and before other code)
                        insert_idx = 0
                        in_docstring = False
                        
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            if stripped.startswith('"""') or stripped.startswith("'''"):
                                in_docstring = not in_docstring
                            elif not in_docstring and (stripped.startswith('import ') or stripped.startswith('from ')):
                                insert_idx = i + 1
                            elif not in_docstring and stripped and not stripped.startswith('#'):
                                break
                        
                        # Insert the import
                        lines.insert(insert_idx, f"import {missing_import}")
                        context.write_file(file_path, '\n'.join(lines))
                        fixes_applied.append(f"{missing_import} in {file_path}")
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Added missing imports: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No import fixes needed"
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Import fix failed: {str(e)}"
            )