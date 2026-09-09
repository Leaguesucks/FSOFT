import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_markdown_plus_latex/flutter_markdown_plus_latex.dart';
import 'package:markdown/markdown.dart' as md;

import 'code_block.dart';

class MessageBubble extends StatelessWidget {
  final String text;
  final bool isUser;

  const MessageBubble({
    super.key,
    required this.text,
    required this.isUser,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Align(
      alignment: isUser
          ? Alignment.centerRight
          : Alignment.centerLeft,

      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.80,
        ),

        margin: const EdgeInsets.only(
          bottom: 12,
        ),

        padding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 12,
        ),

        decoration: BoxDecoration(
          color: isUser
              ? colorScheme.primary
              : colorScheme.surfaceContainerHighest,

          borderRadius: BorderRadius.circular(18),
        ),

        child: SelectionArea(
          child: isUser
              ? Text(
                  text,
                  style: TextStyle(
                    fontSize: 15,
                    height: 1.4,
                    color: colorScheme.onPrimary,
                  ),
                )

              : MarkdownBody(
                  data: text,

                  builders: {
                    'pre': CodeBlockBuilder(),

                    'latex': LatexElementBuilder(
                      textStyle: TextStyle(
                        fontSize: 16,
                        color: colorScheme.onSurface,
                      ),
                    ),
                  },

                  extensionSet: md.ExtensionSet(
                    [
                      LatexBlockSyntax(),
                    ],
                    [
                      LatexInlineSyntax(),
                    ],
                  ),

                  styleSheet: MarkdownStyleSheet(
                    p: TextStyle(
                      fontSize: 15,
                      height: 1.4,
                      color: colorScheme.onSurface,
                    ),

                    h1: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      height: 1.3,
                      color: colorScheme.onSurface,
                    ),

                    h2: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      height: 1.3,
                      color: colorScheme.onSurface,
                    ),

                    h3: TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.bold,
                      height: 1.3,
                      color: colorScheme.onSurface,
                    ),

                    strong: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: colorScheme.onSurface,
                    ),

                    em: TextStyle(
                      fontStyle: FontStyle.italic,
                      color: colorScheme.onSurface,
                    ),

                    listBullet: TextStyle(
                      fontSize: 15,
                      color: colorScheme.onSurface,
                    ),

                    blockquote: TextStyle(
                      fontSize: 15,
                      height: 1.4,
                      color: colorScheme.onSurfaceVariant,
                    ),

                    code: TextStyle(
                      fontSize: 13,
                      fontFamily: 'monospace',
                      color: colorScheme.onSurface,
                      backgroundColor: colorScheme.surface,
                    ),

                    blockSpacing: 10,
                    listIndent: 24,
                  ),
                ),
        ),
      ),
    );
  }
}
