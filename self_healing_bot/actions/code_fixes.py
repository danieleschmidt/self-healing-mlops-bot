"""Code repair actions for common ML pipeline issues."""

import re
import ast
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from .base import BaseAction, ActionResult
from ..core.context import Context
from ..integrations.github import GitHubIntegration

logger = logging.getLogger(__name__)


class CodeFixAction(BaseAction):
    """Apply common code fixes for ML pipeline issues."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.fix_patterns = self.config.get("fix_patterns", self._get_default_patterns())
        self.github_integration = GitHubIntegration()
        self.create_pr = self.config.get("create_pr", True)
        self.pr_branch_prefix = self.config.get("pr_branch_prefix", "bot/code-fix")
        self.backup_enabled = self.config.get("backup_enabled", True)
    
    def can_handle(self, issue_type: str) -> bool:
        return issue_type in [
            "import_error", "syntax_error", "type_error", "gpu_oom", 
            "test_failure", "code_quality", "performance_issue", "dependency_error"
        ]
    
    async def execute(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Execute code fixes based on issue type."""
        issue_type = issue_data.get("type", "")
        
        if not self.can_handle(issue_type):
            return self.create_result(
                success=False,
                message=f"Cannot handle issue type: {issue_type}"
            )
        
        try:
            self.log_action(context, f"Starting {issue_type} fix")
            
            # Create backup if enabled
            backup_data = None
            if self.backup_enabled:
                backup_data = await self._create_backup(context, issue_data)
            
            # Apply appropriate fix
            result = None
            if issue_type == "import_error":
                result = await self._fix_import_error(context, issue_data)
            elif issue_type == "gpu_oom":
                result = await self._fix_gpu_oom(context, issue_data)
            elif issue_type == "syntax_error":
                result = await self._fix_syntax_error(context, issue_data)
            elif issue_type == "type_error":
                result = await self._fix_type_error(context, issue_data)
            elif issue_type == "test_failure":
                result = await self._fix_test_failures(context, issue_data)
            elif issue_type == "code_quality":
                result = await self._fix_code_quality_issues(context, issue_data)
            elif issue_type == "performance_issue":
                result = await self._optimize_performance(context, issue_data)
            elif issue_type == "dependency_error":
                result = await self._fix_dependency_issues(context, issue_data)
            else:
                return self.create_result(
                    success=False,
                    message=f"No specific fix available for {issue_type}"
                )
            
            # Create PR if successful and enabled
            if result.success and self.create_pr:
                pr_result = await self._create_fix_pr(context, issue_data, result.data)
                if pr_result:
                    result.data["pull_request"] = pr_result
            
            # Add backup info to result
            if backup_data:
                result.data["backup"] = backup_data
                
            return result
                
        except Exception as e:
            logger.exception(f"Error applying code fix: {e}")
            # Attempt rollback if backup exists
            if backup_data:
                await self._rollback_changes(context, backup_data)
            
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
        try:
            affected_files = issue_data.get("affected_files", [])
            error_message = issue_data.get("error_message", "")
            fixes_applied = []
            
            for file_path in affected_files:
                content = context.read_file(file_path)
                
                # Try to parse the code and identify type issues
                try:
                    tree = ast.parse(content)
                    modified_content = self._fix_type_annotations(content, tree)
                    
                    if modified_content != content:
                        context.write_file(file_path, modified_content)
                        fixes_applied.append(f"Added type annotations in {file_path}")
                except SyntaxError:
                    # Skip files that can't be parsed
                    continue
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Applied type fixes: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No type error fixes could be applied"
                )
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Type error fix failed: {str(e)}"
            )
    
    async def _fix_test_failures(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix common test failures."""
        try:
            test_files = issue_data.get("test_files", [])
            failure_patterns = issue_data.get("failure_patterns", [])
            fixes_applied = []
            
            for test_file in test_files:
                content = context.read_file(test_file)
                modified = False
                
                # Fix common test issues
                for pattern in failure_patterns:
                    if "assertion" in pattern.lower():
                        content, fixed = self._fix_assertion_errors(content)
                        if fixed:
                            fixes_applied.append(f"Fixed assertion errors in {test_file}")
                            modified = True
                    
                    elif "import" in pattern.lower():
                        content, fixed = self._fix_test_imports(content)
                        if fixed:
                            fixes_applied.append(f"Fixed import issues in {test_file}")
                            modified = True
                    
                    elif "fixture" in pattern.lower():
                        content, fixed = self._fix_pytest_fixtures(content)
                        if fixed:
                            fixes_applied.append(f"Fixed fixture issues in {test_file}")
                            modified = True
                
                if modified:
                    context.write_file(test_file, content)
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Applied test fixes: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No test fixes could be applied"
                )
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Test fix failed: {str(e)}"
            )
    
    async def _fix_code_quality_issues(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix code quality issues."""
        try:
            affected_files = issue_data.get("affected_files", [])
            quality_issues = issue_data.get("quality_issues", [])
            fixes_applied = []
            
            for file_path in affected_files:
                content = context.read_file(file_path)
                original_content = content
                
                # Apply various code quality fixes
                content = self._fix_import_organization(content)
                content = self._fix_code_formatting(content)
                content = self._fix_docstrings(content)
                content = self._remove_unused_imports(content)
                content = self._fix_variable_naming(content)
                
                if content != original_content:
                    context.write_file(file_path, content)
                    fixes_applied.append(f"Improved code quality in {file_path}")
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Applied code quality fixes: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No code quality fixes needed"
                )
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Code quality fix failed: {str(e)}"
            )
    
    async def _optimize_performance(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Optimize code performance."""
        try:
            affected_files = issue_data.get("affected_files", [])
            performance_issues = issue_data.get("performance_issues", [])
            fixes_applied = []
            
            for file_path in affected_files:
                content = context.read_file(file_path)
                original_content = content
                
                # Apply performance optimizations
                content = self._optimize_loops(content)
                content = self._optimize_data_structures(content)
                content = self._add_caching(content)
                content = self._optimize_imports(content)
                
                if content != original_content:
                    context.write_file(file_path, content)
                    fixes_applied.append(f"Optimized performance in {file_path}")
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Applied performance optimizations: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No performance optimizations needed"
                )
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Performance optimization failed: {str(e)}"
            )
    
    async def _fix_dependency_issues(self, context: Context, issue_data: Dict[str, Any]) -> ActionResult:
        """Fix dependency-related issues comprehensively."""
        try:
            missing_deps = issue_data.get("missing_dependencies", [])
            version_conflicts = issue_data.get("version_conflicts", [])
            fixes_applied = []
            
            # Update requirements.txt
            requirements_content = context.read_file("requirements.txt")
            original_requirements = requirements_content
            
            # Add missing dependencies with compatible versions
            for dep in missing_deps:
                if dep not in requirements_content:
                    compatible_version = self._get_compatible_version(dep)
                    requirements_content += f"\n{dep}>={compatible_version}\n"
                    fixes_applied.append(f"Added {dep}>={compatible_version}")
            
            # Resolve version conflicts
            for conflict in version_conflicts:
                package = conflict.get("package", "")
                suggested_version = conflict.get("suggested_version", "")
                if package and suggested_version:
                    requirements_content = re.sub(
                        rf"{package}[>=<!\s\d\.]*",
                        f"{package}=={suggested_version}",
                        requirements_content
                    )
                    fixes_applied.append(f"Updated {package} to {suggested_version}")
            
            if requirements_content != original_requirements:
                context.write_file("requirements.txt", requirements_content)
            
            # Update imports in Python files
            affected_files = issue_data.get("affected_files", [])
            for file_path in affected_files:
                if file_path.endswith(".py"):
                    content = context.read_file(file_path)
                    original_content = content
                    content = self._update_import_statements(content, missing_deps)
                    
                    if content != original_content:
                        context.write_file(file_path, content)
                        fixes_applied.append(f"Updated imports in {file_path}")
            
            if fixes_applied:
                return self.create_result(
                    success=True,
                    message=f"Fixed dependency issues: {', '.join(fixes_applied)}",
                    data={"fixes": fixes_applied}
                )
            else:
                return self.create_result(
                    success=False,
                    message="No dependency fixes needed"
                )
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Dependency fix failed: {str(e)}"
            )
    
    async def _create_backup(self, context: Context, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create backup of files before applying fixes."""
        try:
            affected_files = issue_data.get("affected_files", [])
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "files": {}
            }
            
            for file_path in affected_files:
                try:
                    content = context.read_file(file_path)
                    backup_data["files"][file_path] = content
                except Exception as e:
                    logger.warning(f"Could not backup {file_path}: {e}")
            
            return backup_data
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return {}
    
    async def _rollback_changes(self, context: Context, backup_data: Dict[str, Any]) -> None:
        """Rollback changes using backup data."""
        try:
            files = backup_data.get("files", {})
            for file_path, content in files.items():
                context.write_file(file_path, content)
            logger.info(f"Rolled back changes to {len(files)} files")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    async def _create_fix_pr(self, context: Context, issue_data: Dict[str, Any], fix_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a GitHub pull request for the fixes."""
        try:
            issue_type = issue_data.get("type", "")
            branch_name = f"{self.pr_branch_prefix}-{issue_type}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            
            # Get file changes
            file_changes = context.get_file_changes()
            
            if not file_changes:
                return None
            
            # Create PR title and body
            title = f"🤖 Fix {issue_type.replace('_', ' ').title()} Issues"
            body = self._generate_pr_body(issue_type, fix_data)
            
            # Create pull request
            installation_id = context.get_state("github_installation_id", 1)  # Default for demo
            
            pr_result = await self.github_integration.create_pull_request(
                installation_id=installation_id,
                repo_full_name=context.repo_full_name,
                title=title,
                body=body,
                head_branch=branch_name,
                base_branch="main",
                file_changes=file_changes
            )
            
            return pr_result
            
        except Exception as e:
            logger.error(f"PR creation failed: {e}")
            return None
    
    def _generate_pr_body(self, issue_type: str, fix_data: Dict[str, Any]) -> str:
        """Generate PR body for code fixes."""
        fixes = fix_data.get("fixes", [])
        
        body = f"""## Automated {issue_type.replace('_', ' ').title()} Fixes
        
This PR contains automated fixes for {issue_type} issues detected in the repository.

### Changes Made:
"""
        
        for fix in fixes:
            body += f"- {fix}\n"
        
        body += f"""
### Details:
- **Issue Type**: {issue_type}
- **Fixes Applied**: {len(fixes)}
- **Auto-generated**: Yes
- **Backup Created**: Yes

### Testing:
- [ ] Code compiles without errors
- [ ] Tests pass
- [ ] No regressions introduced

🤖 This PR was automatically generated by the Self-Healing MLOps Bot.
"""
        
        return body
    
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
    
    def _fix_type_annotations(self, content: str, tree: ast.AST) -> str:
        """Add basic type annotations to function definitions."""
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.lineno <= len(lines):
                    line = lines[node.lineno - 1]
                    # Simple type annotation additions for common patterns
                    if '-> ' not in line and 'def ' in line:
                        if line.strip().endswith(':'):
                            # Add return type annotation
                            lines[node.lineno - 1] = line.replace(':', ' -> Any:')
        
        return '\n'.join(lines)
    
    def _fix_assertion_errors(self, content: str) -> Tuple[str, bool]:
        """Fix common assertion patterns in tests."""
        modified = False
        
        # Fix common assertion patterns
        patterns = [
            (r'assert (.+) == True', r'assert \1'),
            (r'assert (.+) == False', r'assert not \1'),
            (r'assert (.+) != None', r'assert \1 is not None'),
            (r'assert (.+) == None', r'assert \1 is None'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
        
        return content, modified
    
    def _fix_test_imports(self, content: str) -> Tuple[str, bool]:
        """Fix common test import issues."""
        modified = False
        
        # Add pytest import if using pytest features
        if 'pytest' in content and 'import pytest' not in content:
            content = 'import pytest\n' + content
            modified = True
        
        # Add unittest import for TestCase usage
        if 'TestCase' in content and 'from unittest import TestCase' not in content:
            content = 'from unittest import TestCase\n' + content
            modified = True
        
        return content, modified
    
    def _fix_pytest_fixtures(self, content: str) -> Tuple[str, bool]:
        """Fix pytest fixture issues."""
        modified = False
        
        # Add @pytest.fixture decorator to functions that look like fixtures
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if (line.strip().startswith('def ') and 
                ('fixture' in line or line.endswith('_fixture():')) and
                i > 0 and '@pytest.fixture' not in lines[i-1]):
                lines.insert(i, '@pytest.fixture')
                modified = True
        
        return '\n'.join(lines), modified
    
    def _fix_import_organization(self, content: str) -> str:
        """Organize imports according to PEP8."""
        lines = content.split('\n')
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in lines:
            if in_imports and (line.strip().startswith('import ') or line.strip().startswith('from ')):
                import_lines.append(line)
            elif line.strip() == '' and in_imports:
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        # Sort imports: standard library, third party, local imports
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        stdlib_modules = {'os', 'sys', 're', 'json', 'time', 'datetime', 'collections', 'typing'}
        
        for line in import_lines:
            if line.strip():
                module = line.split()[1].split('.')[0] if len(line.split()) > 1 else ''
                if module in stdlib_modules:
                    stdlib_imports.append(line)
                elif '.' in module or module.startswith('..'):
                    local_imports.append(line)
                else:
                    third_party_imports.append(line)
        
        organized_imports = []
        if stdlib_imports:
            organized_imports.extend(sorted(stdlib_imports))
            organized_imports.append('')
        if third_party_imports:
            organized_imports.extend(sorted(third_party_imports))
            organized_imports.append('')
        if local_imports:
            organized_imports.extend(sorted(local_imports))
            organized_imports.append('')
        
        return '\n'.join(organized_imports + other_lines)
    
    def _fix_code_formatting(self, content: str) -> str:
        """Apply basic code formatting fixes."""
        # Fix spacing around operators
        content = re.sub(r'([a-zA-Z0-9_])\+([a-zA-Z0-9_])', r'\1 + \2', content)
        content = re.sub(r'([a-zA-Z0-9_])-([a-zA-Z0-9_])', r'\1 - \2', content)
        content = re.sub(r'([a-zA-Z0-9_])\*([a-zA-Z0-9_])', r'\1 * \2', content)
        content = re.sub(r'([a-zA-Z0-9_])/([a-zA-Z0-9_])', r'\1 / \2', content)
        
        # Fix spacing around commas
        content = re.sub(r',([^\s])', r', \1', content)
        
        return content
    
    def _fix_docstrings(self, content: str) -> str:
        """Add basic docstrings to functions without them."""
        lines = content.split('\n')
        result_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            result_lines.append(line)
            
            # Check if this is a function definition without a docstring
            if (line.strip().startswith('def ') and 
                line.strip().endswith(':') and 
                i + 1 < len(lines) and 
                not lines[i + 1].strip().startswith('"""') and
                not lines[i + 1].strip().startswith("'''")):
                
                # Add a basic docstring
                indent = len(line) - len(line.lstrip())
                result_lines.append(' ' * (indent + 4) + '"""TODO: Add function description."""')
            
            i += 1
        
        return '\n'.join(result_lines)
    
    def _remove_unused_imports(self, content: str) -> str:
        """Remove obviously unused imports."""
        lines = content.split('\n')
        import_lines = []
        code_lines = []
        
        # Separate imports from code
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append(line)
            else:
                code_lines.append(line)
        
        code_content = '\n'.join(code_lines)
        filtered_imports = []
        
        for import_line in import_lines:
            # Extract module name
            if import_line.strip().startswith('import '):
                module = import_line.split()[1].split('.')[0]
            elif import_line.strip().startswith('from '):
                parts = import_line.split()
                if len(parts) > 3:
                    module = parts[3]
                else:
                    module = parts[1]
            else:
                continue
            
            # Keep import if module is used in code
            if module in code_content:
                filtered_imports.append(import_line)
        
        return '\n'.join(filtered_imports + [''] + code_lines)
    
    def _fix_variable_naming(self, content: str) -> str:
        """Fix common variable naming issues."""
        # Convert camelCase to snake_case for variables
        content = re.sub(r'\b([a-z]+)([A-Z][a-z]+)([A-Z][a-z]+)*\b', 
                        lambda m: '_'.join(m.group().split()), content)
        
        return content
    
    def _optimize_loops(self, content: str) -> str:
        """Apply basic loop optimizations."""
        # Replace list comprehensions where appropriate
        content = re.sub(
            r'for (.+) in (.+):\s*\n\s*(.+)\.append\((.+)\)',
            r'\3 = [\4 for \1 in \2]',
            content
        )
        
        return content
    
    def _optimize_data_structures(self, content: str) -> str:
        """Optimize data structure usage."""
        # Suggest set for membership testing
        content = re.sub(
            r'if (.+) in \[(.+)\]:',
            r'if \1 in {\2}:',
            content
        )
        
        return content
    
    def _add_caching(self, content: str) -> str:
        """Add caching decorators where appropriate."""
        # Add lru_cache to pure functions that look expensive
        if 'from functools import lru_cache' not in content:
            content = 'from functools import lru_cache\n' + content
        
        lines = content.split('\n')
        result_lines = []
        
        for i, line in enumerate(lines):
            if (line.strip().startswith('def ') and 
                'cache' not in line and
                i > 0 and '@' not in lines[i-1]):
                
                # Add cache decorator to functions that look cacheable
                if any(keyword in line for keyword in ['calculate', 'compute', 'process']):
                    indent = len(line) - len(line.lstrip())
                    result_lines.append(' ' * indent + '@lru_cache(maxsize=128)')
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _optimize_imports(self, content: str) -> str:
        """Optimize import statements for performance."""
        # Convert wildcard imports to specific imports where possible
        content = re.sub(r'from (.+) import \*', r'# TODO: Replace wildcard import from \1', content)
        
        return content
    
    def _get_compatible_version(self, package: str) -> str:
        """Get compatible version for a package."""
        # Common package versions - in production this would query PyPI
        version_map = {
            'torch': '1.12.0',
            'tensorflow': '2.8.0',
            'numpy': '1.21.0',
            'pandas': '1.3.0',
            'scikit-learn': '1.0.0',
            'matplotlib': '3.5.0',
            'seaborn': '0.11.0',
            'requests': '2.27.0',
            'flask': '2.0.0',
            'fastapi': '0.75.0'
        }
        
        return version_map.get(package, '0.1.0')
    
    def _update_import_statements(self, content: str, missing_deps: List[str]) -> str:
        """Update import statements to use newly added dependencies."""
        for dep in missing_deps:
            # Add import statement if not present
            if dep not in content and not any(line.strip().startswith(f'import {dep}') or 
                                            line.strip().startswith(f'from {dep}') 
                                            for line in content.split('\n')):
                content = f'import {dep}\n' + content
        
        return content


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