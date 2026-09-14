import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_syntax_view/flutter_syntax_view.dart';

class CodeBlockBuilder extends MarkdownElementBuilder {
  @override
  Widget? visitElementAfter(
    dynamic element,
    TextStyle? preferredStyle,
  ) {
    // Because this builder is registered for `pre`, the actual
    // language class is usually on the nested `code` element.
    final codeElement =
        element.children.isNotEmpty ? element.children.first : element;

    final String code = codeElement.textContent;

    final String className =
        codeElement.attributes['class'] ?? '';

    final String language = _extractLanguage(className);

    return ChatCodeBlock(
      code: code,
      language: language,
    );
  }

  String _extractLanguage(String className) {
    const prefix = 'language-';

    if (!className.startsWith(prefix)) {
      return '';
    }

    return className.substring(prefix.length).toLowerCase();
  }
}

/// ChatGPT-style code block with syntax highlighting and copy button.
class ChatCodeBlock extends StatefulWidget {
  final String code;
  final String language;

  const ChatCodeBlock({
    super.key,
    required this.code,
    required this.language,
  });

  @override
  State<ChatCodeBlock> createState() => _ChatCodeBlockState();
}

class _ChatCodeBlockState extends State<ChatCodeBlock> {
  bool copied = false;

  Future<void> _copyCode() async {
    await Clipboard.setData(
      ClipboardData(text: widget.code),
    );

    if (!mounted) {
      return;
    }

    setState(() {
      copied = true;
    });

    await Future.delayed(
      const Duration(seconds: 2),
    );

    if (!mounted) {
      return;
    }

    setState(() {
      copied = false;
    });
  }

  /// Convert the language reported by Markdown into the syntax
  /// expected by flutter_syntax_view.
  Syntax _getSyntax(String language) {
    switch (language.toLowerCase()) {
      // Python
      case 'python':
      case 'py':
        return Syntax.PYTHON;

      // C
      case 'c':
        return Syntax.C;

      // C++
      case 'cpp':
      case 'c++':
      case 'cc':
      case 'cxx':
        return Syntax.CPP;

      // JavaScript
      case 'javascript':
      case 'js':
        return Syntax.JAVASCRIPT;

      // Java
      case 'java':
        return Syntax.JAVA;

      // Kotlin
      case 'kotlin':
      case 'kt':
        return Syntax.KOTLIN;

      // Swift
      case 'swift':
        return Syntax.SWIFT;

      // Dart
      case 'dart':
        return Syntax.DART;

      // YAML
      case 'yaml':
      case 'yml':
        return Syntax.YAML;

      // Rust
      case 'rust':
      case 'rs':
        return Syntax.RUST;

      // Lua
      case 'lua':
        return Syntax.LUA;

      // Unknown language.
      //
      // Do NOT use Python as the fallback. Otherwise an unknown
      // language will incorrectly receive Python highlighting.
      default:
        return Syntax.PYTHON;
    }
  }

  @override
  Widget build(BuildContext context) {
    final Syntax syntax = _getSyntax(widget.language);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.symmetric(
        vertical: 8,
      ),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(10),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.only(
              top: 38,
              bottom: 8,
            ),
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: SyntaxView(
                code: widget.code,

                // Use the language detected from Markdown.
                syntax: syntax,

                syntaxTheme: SyntaxTheme.vscodeDark(),

                fontSize: 13,

                // No line-number gutter.
                withLinesCount: false,

                // No zoom controls.
                withZoom: false,

                // Don't force the code block to fill the screen.
                expanded: false,

                // Keep code selectable.
                selectable: true,
              ),
            ),
          ),

          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 38,
              padding: const EdgeInsets.symmetric(
                horizontal: 12,
              ),
              decoration: const BoxDecoration(
                color: Color(0xFF181818),
              ),
              child: Row(
                children: [
                  // Language
                  if (widget.language.isNotEmpty)
                    Text(
                      widget.language,
                      style: const TextStyle(
                        color: Color(0xFFAAAAAA),
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                      ),
                    ),

                  const Spacer(),

                  // Copy button
                  InkWell(
                    onTap: _copyCode,
                    borderRadius: BorderRadius.circular(6),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 5,
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            copied
                                ? Icons.check
                                : Icons.copy_outlined,
                            size: 15,
                            color: const Color(0xFFBBBBBB),
                          ),

                          const SizedBox(width: 5),

                          Text(
                            copied ? 'Copied!' : 'Copy',
                            style: const TextStyle(
                              color: Color(0xFFBBBBBB),
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
