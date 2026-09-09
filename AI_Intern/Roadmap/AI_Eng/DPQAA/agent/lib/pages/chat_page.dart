import 'package:flutter/material.dart';

import '../services/chat_service.dart';
import '../widgets/message_bubble.dart';
import '../widgets/thinking_bubble.dart';
import '../widgets/chat_input.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final List<Map<String, dynamic>> _messages = [
    {
      "text": "Hello! How can I assist you today?",
      "isUser": false,
    },
  ];

  final TextEditingController _textEditingController =
      TextEditingController();

  final ScrollController _scrollController =
      ScrollController();

  final ChatService _chatService = ChatService();

  bool _isThinking = false;

  String _thinkingStatus = "Thinking...";

  Future<void> _sendMessage() async {
    final text = _textEditingController.text.trim();

    if (text.isEmpty || _isThinking) {
      return;
    }

    setState(() {
      _messages.add({
        "text": text,
        "isUser": true,
      });

      _isThinking = true;
      _thinkingStatus = "Resolving query...";
    });

    _textEditingController.clear();

    _scrollToBottom();

    try {
      bool receivedFirstChunk = false;

      await _chatService.sendMessage(
        text,

        onStatus: (status) {
          if (!mounted) {
            return;
          }

          setState(() {
            _thinkingStatus = status;
          });

          _scrollToBottom();
        },

        onChunk: (chunk) {
          if (!mounted) {
            return;
          }

          setState(() {
            if (!receivedFirstChunk) {
              receivedFirstChunk = true;

              _isThinking = false;

              _messages.add({
                "text": chunk,
                "isUser": false,
              });
            } else {
              _messages.last["text"] += chunk;
            }
          });

          _scrollToBottom();
        },
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _isThinking = false;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        _isThinking = false;

        _messages.add({
          "text": "Sorry, I couldn't connect to the server.",
          "isUser": false,
        });
      });

      debugPrint(e.toString());
    }
  }

  void _scrollToBottom() {
    Future.delayed(
      const Duration(milliseconds: 50),
      () {
        if (!_scrollController.hasClients) {
          return;
        }

        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            CircleAvatar(
              radius: 18,
              child: ClipOval(
                child: Image.asset(
                  'assets/images/fsoft_logo.png',
                  width: 28,
                  height: 28,
                  fit: BoxFit.contain,
                ),
              ),
            ),

            const SizedBox(width: 12),

            const Text(
              'OUR FPT AI Assistant',
              style: TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),

        actions: [
          IconButton(
            icon: const Icon(Icons.more_vert),
            onPressed: () {
              // TODO: Settings
            },
          ),
        ],
      ),

      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,

              padding: const EdgeInsets.symmetric(
                vertical: 10,
                horizontal: 16,
              ),

              itemCount:
                  _messages.length + (_isThinking ? 1 : 0),

              itemBuilder: (context, index) {
                if (_isThinking &&
                    index == _messages.length) {
                  return ThinkingBubble(
                    message: _thinkingStatus,
                  );
                }

                final message = _messages[index];

                return MessageBubble(
                  text: message["text"],
                  isUser: message["isUser"],
                );
              },
            ),
          ),

          ChatInput(
            controller: _textEditingController,
            isThinking: _isThinking,
            onSend: _sendMessage,
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _textEditingController.dispose();
    _scrollController.dispose();
    _chatService.dispose();

    super.dispose();
  }
}
