import 'dart:math' as math;

import 'package:flutter/material.dart';

class ThinkingBubble extends StatefulWidget {
  final String message;

  const ThinkingBubble({
    super.key,
    required this.message,
  });

  @override
  State<ThinkingBubble> createState() => _ThinkingBubbleState();
}

class _ThinkingBubbleState extends State<ThinkingBubble>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(
        milliseconds: 2500,
      ),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Align(
      alignment: Alignment.centerLeft,

      child: Padding(
        padding: const EdgeInsets.only(
          left: 16,
          bottom: 12,
        ),

        child: AnimatedSwitcher(
          duration: const Duration(
            milliseconds: 250,
          ),

          transitionBuilder: (
            Widget child,
            Animation<double> animation,
          ) {
            return FadeTransition(
              opacity: animation,
              child: child,
            );
          },

          child: AnimatedBuilder(
            key: ValueKey(widget.message),
            animation: _controller,

            builder: (context, child) {
              return Wrap(
                children: List.generate(
                  widget.message.length,
                  (index) {
                    final character = widget.message[index];

                    // Each character gets a slightly different
                    // position in the animation cycle.
                    final phase =
                        (_controller.value +
                                index * 0.055) %
                            1.0;

                    // Smooth wave between -1 and 1.
                    final wave = math.sin(
                      phase * 2 * math.pi,
                    );

                    // Only move upward/downward slightly.
                    final offset = wave * 2.5;

                    return Transform.translate(
                      offset: Offset(
                        0,
                        -offset,
                      ),

                      child: Text(
                        character,

                        style: TextStyle(
                          fontSize: 14,
                          fontStyle: FontStyle.italic,
                          color: colorScheme.onSurface
                              .withValues(alpha: 0.45),
                        ),
                      ),
                    );
                  },
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}