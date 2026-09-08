import 'package:flutter_syntax_view/flutter_syntax_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

/// ChatGPT-style code block.
class CodeBlockBuilder extends MarkdownElementBuilder {
  @override
  Widget? visitElementAfter(
    dynamic element,
    TextStyle? preferredStyle,
  ) {
    final String code = element.textContent;

    final String className =
        element.attributes['class'] ?? '';

    final String language =
        _extractLanguage(className);

    return ChatCodeBlock(
      code: code,
      language: language,
    );
  }

  String _extractLanguage(String className) {
    if (!className.startsWith('language-')) {
      return '';
    }

    return className
        .substring('language-'.length)
        .toLowerCase();
  }
}


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
      ClipboardData(
        text: widget.code,
      ),
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

  Syntax _getSyntax() {
    switch (widget.language) {
      case 'python':
      case 'py':
        return Syntax.PYTHON;

      case 'cpp':
      case 'c++':
        return Syntax.CPP;

      case 'c':
        return Syntax.C;

      case 'javascript':
      case 'js':
        return Syntax.JAVASCRIPT;

      case 'java':
        return Syntax.JAVA;

      case 'kotlin':
      case 'kt':
        return Syntax.KOTLIN;

      case 'swift':
        return Syntax.SWIFT;

      case 'dart':
        return Syntax.DART;

      case 'yaml':
      case 'yml':
        return Syntax.YAML;

      case 'rust':
      case 'rs':
        return Syntax.RUST;

      case 'lua':
        return Syntax.LUA;

      default:
        return Syntax.PYTHON;
    }
  }

  @override
  Widget build(BuildContext context) {
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
          // Code
          Padding(
            padding: const EdgeInsets.only(
              top: 38,
              left: 0,
              right: 0,
              bottom: 8,
            ),
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: SyntaxView(
                code: widget.code,
                syntax: _getSyntax(),
                syntaxTheme: SyntaxTheme.vscodeDark(),
                fontSize: 13,

                // We don't want the line-number gutter.
                withLinesCount: false,

                // No zoom controls.
                withZoom: false,

                // Don't force the block to fill the screen.
                expanded: false,

                // Code remains selectable.
                selectable: true,
              ),
            ),
          ),

          // Header
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