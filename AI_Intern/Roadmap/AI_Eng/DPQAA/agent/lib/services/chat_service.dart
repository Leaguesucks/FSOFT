import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class ChatService {
  final String baseUrl = "http://127.0.0.1:8000";

  final http.Client _client = http.Client();

  Future<void> sendMessage(
    String text, {
      required String userId,
      required String sessionId,
  
      Function(String status)? onStatus,
      Function(String chunk)? onChunk,
      Function() ? onDone,
  }) async {
    final request = http.Request(
      "POST",
      Uri.parse("$baseUrl/chat"),
    );

    request.headers["Content-Type"] = "application/json";

    request.body = jsonEncode({
      "text": text,
      "chat_type": "plain_text",
      "user_id": userId,
      "session_id": sessionId,
    });

    final response = await _client.send(request);

    if (response.statusCode != 200) {
      final body = await response.stream.bytesToString();

      throw Exception(
        "Server returned ${response.statusCode}: $body",
      );
    }

    final stream = response.stream
        .transform(utf8.decoder)
        .transform(const LineSplitter());

    await for (final line in stream) {
      if (line.trim().isEmpty) {
        continue;
      }

      try {
        final event = jsonDecode(line);

        final type = event["type"];

        if (type == "status") {
          final message = event["message"];

          if (message is String) {
            onStatus?.call(message);
          }
        } else if (type == "content") {
          final content = event["content"];

          if (content is String) {
            onChunk?.call(content);
          }
        } else if (type == "done") {
          onDone?.call();
        }
      } catch (e) {
        debugPrint("Failed to parse server event: $line");
        debugPrint(e.toString());
      }
    }
  }

  void dispose() {
    _client.close();
  }
}