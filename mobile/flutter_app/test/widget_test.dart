import 'package:flutter_test/flutter_test.dart';
import 'package:rose_grower/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const RoseGrowerApp());
    expect(find.text('Ekspert od Róż'), findsOneWidget);
  });
}