# 🤝 Contributing to Stock Portfolio Dashboard

Thank you for your interest in contributing to our Stock Portfolio Dashboard! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## 🚀 How Can I Contribute?

### Reporting Bugs
- Use the GitHub issue template
- Provide detailed reproduction steps
- Include system information and error logs
- Check if the bug has already been reported

### Suggesting Enhancements
- Use the feature request template
- Explain the problem and proposed solution
- Consider the impact on existing functionality
- Provide mockups or examples if applicable

### Code Contributions
- Fix bugs or implement features
- Improve documentation
- Add tests for better coverage
- Optimize performance

## 🛠️ Development Setup

### Prerequisites
- Node.js 18.17+
- Python 3.12+
- PostgreSQL 14+
- Redis 6+
- Git

### Local Development
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/stock-portfolio-analyser.git
cd stock-portfolio-analyser

# Install dependencies
npm install
cd python_backend && pip install -r requirements.txt && cd ..

# Set up environment variables
cp .env.example .env
cp python_backend/.env.example python_backend/.env

# Set up database
npx prisma migrate dev
npx prisma generate

# Start development servers
npm run dev
# In another terminal: cd python_backend && uvicorn main:app --reload
```

## 📝 Coding Standards

### TypeScript/JavaScript
- Use **TypeScript** for all new code
- Follow **ESLint** and **Prettier** configurations
- Use meaningful variable and function names
- Add JSDoc comments for complex functions
- Prefer functional components with hooks

### Python
- Follow **PEP 8** style guidelines
- Use type hints for function parameters
- Add docstrings for all functions and classes
- Use meaningful variable names
- Keep functions focused and small

### General
- Write self-documenting code
- Use consistent naming conventions
- Keep functions under 50 lines
- Avoid deep nesting (max 3 levels)
- Use meaningful commit messages

## 🧪 Testing Guidelines

### Frontend Testing
```bash
# Run unit tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

### Backend Testing
```bash
cd python_backend

# Run all tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_forecast.py
```

### Test Requirements
- **Unit tests**: Required for all new functions
- **Integration tests**: Required for API endpoints
- **Test coverage**: Maintain >80% coverage
- **Test naming**: Descriptive test names
- **Mock external services**: Don't hit real APIs in tests

## 🔄 Pull Request Process

### Before Submitting
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes following coding standards
4. **Test** your changes thoroughly
5. **Update** documentation if needed

### Pull Request Guidelines
- **Title**: Clear, descriptive title
- **Description**: Detailed explanation of changes
- **Related issues**: Link to relevant issues
- **Screenshots**: Include for UI changes
- **Testing**: Describe how you tested

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots
Add screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 Reporting Bugs

### Bug Report Template
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 12.0]
- Browser: [e.g., Chrome 96.0]
- Version: [e.g., 1.2.3]

## Additional Information
Screenshots, error logs, etc.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Problem Statement
Clear description of the problem

## Proposed Solution
Description of the proposed solution

## Alternative Solutions
Other approaches considered

## Additional Context
Screenshots, mockups, examples
```

## 📚 Documentation

### Code Documentation
- **JSDoc** for JavaScript/TypeScript functions
- **Docstrings** for Python functions and classes
- **README updates** for new features
- **API documentation** for new endpoints

### User Documentation
- **User guides** for new features
- **Screenshots** and examples
- **Video tutorials** for complex features
- **FAQ updates** based on common issues

## 🔍 Code Review Process

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed
- [ ] Error handling is appropriate

### Review Guidelines
- **Be constructive** and helpful
- **Focus on code quality** and functionality
- **Ask questions** when something is unclear
- **Suggest improvements** when possible
- **Approve** when requirements are met

## 🚀 Release Process

### Versioning
We use [Semantic Versioning](https://semver.org/):
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] Deployment successful

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Pull Requests**: For code reviews and discussions

### Resources
- [Project Wiki](https://github.com/arjanbathla/stock-portfolio-analyser/wiki)
- [API Documentation](http://localhost:8000/docs)
- [Code Style Guide](STYLE_GUIDE.md)

## 🙏 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page
- **Project documentation**

---

**Thank you for contributing to Stock Portfolio Dashboard!** 🎉

Your contributions help make this project better for everyone in the financial technology community. 