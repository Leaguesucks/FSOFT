import 'package:flutter/material.dart';

class ChatInput extends StatelessWidget {
  final TextEditingController controller;
  final bool isThinking;
  final VoidCallback onSend;

  const ChatInput({
    super.key,
    required this.controller,
    required this.isThinking,
    required this.onSend,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(
          12,
          8,
          12,
          12,
        ),

        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: controller,

                onSubmitted: (_) {
                  onSend();
                },

                decoration: InputDecoration(
                  hintText: "Ask about FPT policies...",

                  border: OutlineInputBorder(
                    borderRadius:
                        BorderRadius.circular(25),
                  ),

                  contentPadding:
                      const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 12,
                  ),
                ),
              ),
            ),

            const SizedBox(width: 8),

            IconButton(
              style: IconButton.styleFrom(
                backgroundColor:
                    Theme.of(context)
                        .colorScheme
                        .primary,

                foregroundColor:
                    Theme.of(context)
                        .colorScheme
                        .onPrimary,

                minimumSize: const Size(50, 50),
              ),

              icon: Icon(
                isThinking
                    ? Icons.hourglass_empty
                    : Icons.send,
              ),

              onPressed:
                  isThinking ? null : onSend,
            ),
          ],
        ),
      ),
    );
  }
}